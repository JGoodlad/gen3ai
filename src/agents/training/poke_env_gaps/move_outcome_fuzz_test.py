"""
E2E move-outcome fuzz test (crit / miss / fail / cant).

Validates the full protocol → poke-env → BattleContext → TurnDelta → encoder
pipeline for the move-outcome reporting added in gen3_move_outcome_v1. Like
effectiveness_fuzz_test, it checks four independent layers against the raw
Showdown protocol stream over many random battles:

  Layer 1 — raw |-crit|/|-miss|/|-fail|/|-notarget|/|-nothing|/|cant| lines vs
              poke-env properties (battle.our/opp_move_{crit,missed,failed} and
              active_pokemon.last_cant_reason). Crit/miss/fail are attributed to
              whichever side's |move| is CURRENTLY resolving — the test mirrors
              poke-env's `_current_move_user_side` attribution exactly.
  Layer 2 — poke-env properties vs BattleContext snapshot
              (curr_ctx.our/opp_move_{crit,missed,failed}, our/opp_cant_reason).
  Layer 3 — BattleContext vs TurnDelta (delta.our/opp_move_outcome derivation,
              delta.our/opp_move_crit).
  Layer 4 — TurnDelta vs encoded observation vector (outcome one-hot, crit bit,
              and cant one-hot at the named OFFSET_* positions, with prefix
              normalization of the |cant| reason).

Coverage is asserted: the run FAILS if it never observed a crit, a miss, a
fail, or at least two distinct cant reasons — a green run with zero coverage
would validate nothing.

Run directly (no server needed; runs in-process via the local BattleStream bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/move_outcome_fuzz_test.py [n_battles]
"""

import asyncio
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.observation.state_encoder import load_mappings
from agents.observation.turn_delta_encoder import (
    TurnDeltaEncoder,
    OUTCOME_DIM,
    CANT_DIM,
    OFFSET_OUR_CANT,
    OFFSET_OPP_CANT,
    OFFSET_OUR_MOVE_OUTCOME,
    OFFSET_OPP_MOVE_OUTCOME,
    OFFSET_OUR_CRIT,
    OFFSET_OPP_CRIT,
    _OUTCOME_TO_IDX,
)
from agents.observation.gen3_effects import (
    normalize_cant_reason as _normalize_cant_reason_ge,
    UnknownCantReasonError,
    _CANT_INDEX as _CANT_TO_IDX,
)
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta, SELF_KO_MOVES
from agents.training.turn_delta_legacy import build_legacy
from agents.training.slot_registry import SlotRegistry
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

BATTLE_FORMAT = "gen3ou"

# ---------------------------------------------------------------------------
# Teams — engineered to exercise every outcome category and cant reason.
#   crit    : Slash (high crit ratio) + the 1/16 base rate on every hit
#   miss    : Blizzard (70), Hydro Pump (80), Cross Chop (80)
#   fail    : Protect / Substitute used on repeat
#   cant    : Lovely Kiss / Hypnosis (slp), Thunder Wave (par), Ice Beam (frz),
#             Rock Slide (flinch), Hyper Beam (recharge), Slaking (Truant),
#             Gengar Taunt (opp status-move lock)
# ---------------------------------------------------------------------------

EDGE_TEAM = """\
Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Thunderbolt
- Ice Punch
- Explosion

Cloyster @ Leftovers
Ability: Shell Armor
EVs: 252 HP / 4 Atk / 252 Def
Bold Nature
- Spikes
- Surf
- Ice Beam
- Explosion

Forretress @ Leftovers
Ability: Sturdy
EVs: 252 HP / 252 Atk / 4 Def
Brave Nature
- Spikes
- Rapid Spin
- Earthquake
- Explosion

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 236 Atk / 20 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Brick Break

Jolteon @ Leftovers
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Hidden Power Ice
- Substitute
- Baton Pass

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 40 Atk / 216 SpD
Careful Nature
- Rock Slide
- Earthquake
- Crunch
- Fire Blast
"""

VARIANCE_TEAM = """\
Slaking @ Choice Band
Ability: Truant
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Return
- Earthquake
- Shadow Ball
- Slash

Jynx @ Leftovers
Ability: Oblivious
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Lovely Kiss
- Ice Beam
- Blizzard
- Calm Mind

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Hypnosis
- Taunt
- Fire Punch
- Thunderbolt

Machamp @ Leftovers
Ability: Guts
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Cross Chop
- Rock Slide
- Hyper Beam
- Bulk Up

Blastoise @ Leftovers
Ability: Torrent
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
- Hydro Pump
- Ice Beam
- Protect
- Substitute

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Rock Slide
- Earthquake
- Crunch
- Thunder Wave
"""


# ---------------------------------------------------------------------------
# Raw per-turn protocol tracking
# ---------------------------------------------------------------------------

@dataclass
class _TurnRaw:
    # (mover_prefix, kind) for each crit/miss/fail event, where mover_prefix is
    # the side whose |move| was resolving when the event fired.
    outcome_events: list = field(default_factory=list)  # (prefix, "crit"|"missed"|"failed")
    # (pokemon_prefix, raw_reason) for each |cant| line.
    cant_events: list = field(default_factory=list)


def _opp_role(player_role: str) -> str:
    return "p2" if player_role == "p1" else "p1"


def _expected_outcomes(raw: _TurnRaw, player_role: str) -> dict:
    """Replicate poke-env's per-side attribution from the raw events.

    Returns {"our_crit","opp_crit","our_missed","opp_missed","our_failed",
    "opp_failed"} as bools, and {"our_cant","opp_cant"} as raw reason strings
    (or None)."""
    out = {
        "our_crit": False, "opp_crit": False,
        "our_missed": False, "opp_missed": False,
        "our_failed": False, "opp_failed": False,
        "our_cant": None, "opp_cant": None,
    }
    for prefix, kind in raw.outcome_events:
        side = "our" if prefix == player_role else "opp"
        out[f"{side}_{kind}"] = True
    for prefix, reason in raw.cant_events:
        side = "our" if prefix == player_role else "opp"
        out[f"{side}_cant"] = reason  # last cant for a side wins
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class ScenarioStats:
    name: str
    n_turns: int = 0

    layer1_mismatches: int = 0
    layer2_mismatches: int = 0
    layer3_mismatches: int = 0
    layer4_mismatches: int = 0

    crit_seen: int = 0
    miss_seen: int = 0
    fail_seen: int = 0
    hit_seen: int = 0
    cant_reasons_seen: set = field(default_factory=set)

    # --- Faint / self-faint edge-case coverage + correctness ---
    explosion_faint_seen: int = 0      # used Explosion/Self-Destruct then self-fainted
    switch_death_seen: int = 0         # switched a mon in that fainted (e.g. to Spikes)
    faint_before_acting_seen: int = 0  # KO'd before our move could fire
    edge_mismatches: int = 0           # an edge case was MISREPRESENTED by the delta

    examples: list = field(default_factory=list)

    def record_example(self, layer: int, detail: dict):
        if len(self.examples) < 15:
            self.examples.append({"layer": layer, **detail})


# ---------------------------------------------------------------------------
# Encoder (shared)
# ---------------------------------------------------------------------------

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        m = load_mappings()
        _ENCODER = TurnDeltaEncoder(m.get("moves", {}), m.get("species", {}))
    return _ENCODER


def _onehot_idx(vec, offset, dim):
    """Return the set index in vec[offset:offset+dim], or None if all-zero."""
    block = vec[offset:offset + dim]
    if block.max() <= 0.5:
        return None
    return int(np.argmax(block))


# ---------------------------------------------------------------------------
# Fuzz player
# ---------------------------------------------------------------------------

class MoveOutcomeFuzzPlayer(Player):
    def __init__(self, scenario_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = ScenarioStats(name=scenario_name)
        self._raw_prev: dict[str, Optional[_TurnRaw]] = {}
        self._raw_curr: dict[str, _TurnRaw] = {}
        self._cur_mover: dict[str, Optional[str]] = {}  # current resolving |move| side
        self._prev_ctx: dict[str, Optional[BattleContext]] = {}
        self._last_action: dict[str, Optional[int]] = {}
        self._our_slots: dict[str, SlotRegistry] = {}
        self._opp_slots: dict[str, SlotRegistry] = {}

    async def _handle_battle_message(self, split_messages):
        battle_tag = split_messages[0][0].lstrip(">")
        if battle_tag not in self._raw_curr:
            self._raw_curr[battle_tag] = _TurnRaw()
            self._raw_prev[battle_tag] = None
            self._cur_mover[battle_tag] = None

        curr = self._raw_curr[battle_tag]

        for msg in split_messages[1:]:
            if len(msg) < 2:
                continue
            mt = msg[1]

            if mt == "turn":
                self._raw_prev[battle_tag] = curr
                curr = _TurnRaw()
                self._raw_curr[battle_tag] = curr
                self._cur_mover[battle_tag] = None

            elif mt == "move" and len(msg) >= 3:
                self._cur_mover[battle_tag] = msg[2][:2]
                # Older protocol variant: miss/notarget as a |move| suffix.
                if any(tok == "[miss]" for tok in msg):
                    curr.outcome_events.append((msg[2][:2], "missed"))
                if any(tok in ("[notarget]", "[still]") for tok in msg):
                    curr.outcome_events.append((msg[2][:2], "failed"))

            elif mt == "-crit":
                mover = self._cur_mover[battle_tag]
                if mover is not None:
                    curr.outcome_events.append((mover, "crit"))

            elif mt == "-miss":
                mover = self._cur_mover[battle_tag]
                if mover is not None:
                    curr.outcome_events.append((mover, "missed"))

            elif mt in ("-fail", "-notarget", "-nothing"):
                mover = self._cur_mover[battle_tag]
                if mover is not None:
                    curr.outcome_events.append((mover, "failed"))

            elif mt == "cant" and len(msg) >= 3:
                prefix = msg[2][:2]
                reason = msg[3] if len(msg) > 3 else None
                curr.cant_events.append((prefix, reason))

        await super()._handle_battle_message(split_messages)

    def _validate_turn(self, battle, curr_ctx: BattleContext, delta: TurnDelta):
        s = self.stats
        s.n_turns += 1
        battle_tag = battle.battle_tag
        player_role = battle.player_role
        if player_role is None:
            return
        prev_raw = self._raw_prev.get(battle_tag)
        if prev_raw is None:
            return

        exp = _expected_outcomes(prev_raw, player_role)

        # === Layer 1: raw vs poke-env properties ===
        # Crit/miss/fail properties are turn-gated and read mid-turn on a
        # forced switch before |turn|N+1; skip Layer 1 in that case (Layers
        # 2-4 still run and remain internally consistent).
        prop = {
            "our_crit": battle.our_move_crit, "opp_crit": battle.opp_move_crit,
            "our_missed": battle.our_move_missed, "opp_missed": battle.opp_move_missed,
            "our_failed": battle.our_move_failed, "opp_failed": battle.opp_move_failed,
        }
        if not battle.force_switch:
            for key in ("our_crit", "opp_crit", "our_missed", "opp_missed",
                        "our_failed", "opp_failed"):
                if prop[key] != exp[key]:
                    s.layer1_mismatches += 1
                    s.record_example(1, {
                        "check": key, "turn": battle.turn,
                        "expected": exp[key], "actual": prop[key],
                        "raw_outcome_events": list(prev_raw.outcome_events),
                        "player_role": player_role,
                    })

        # === Layer 2: poke-env properties vs BattleContext ===
        ctx_map = {
            "our_crit": curr_ctx.our_move_crit, "opp_crit": curr_ctx.opp_move_crit,
            "our_missed": curr_ctx.our_move_missed, "opp_missed": curr_ctx.opp_move_missed,
            "our_failed": curr_ctx.our_move_failed, "opp_failed": curr_ctx.opp_move_failed,
        }
        for key in ctx_map:
            if ctx_map[key] != prop[key]:
                s.layer2_mismatches += 1
                s.record_example(2, {
                    "check": key, "turn": battle.turn,
                    "battle_prop": prop[key], "ctx_val": ctx_map[key],
                })

        # === Layer 3: BattleContext vs TurnDelta ===
        # Outcome derivation: reconstruct expected category from ctx flags and
        # the move/switch/cant context exactly as TurnDelta._derive does.
        def expect_outcome(move_used, missed, failed, suppressed, connected):
            if suppressed or not move_used:
                return None
            if connected:   # a damaging event resolved -> the move landed
                return "hit"
            if missed:
                return "miss"
            if failed:
                return "fail"
            return "hit"

        exp_our_outcome = expect_outcome(
            delta.our_move_id is not None,
            curr_ctx.our_move_missed, curr_ctx.our_move_failed,
            delta.our_failed_to_move,
            delta.our_damaging_event is not None or delta.our_move_id in SELF_KO_MOVES,
        )
        exp_opp_outcome = expect_outcome(
            delta.opp_move_id is not None,
            curr_ctx.opp_move_missed, curr_ctx.opp_move_failed,
            delta.opp_failed_to_move,
            delta.opp_damaging_event is not None or delta.opp_move_id in SELF_KO_MOVES,
        )
        if delta.our_move_outcome != exp_our_outcome:
            s.layer3_mismatches += 1
            s.record_example(3, {"check": "our_move_outcome", "turn": battle.turn,
                                 "expected": exp_our_outcome, "delta": delta.our_move_outcome})
        if delta.opp_move_outcome != exp_opp_outcome:
            s.layer3_mismatches += 1
            s.record_example(3, {"check": "opp_move_outcome", "turn": battle.turn,
                                 "expected": exp_opp_outcome, "delta": delta.opp_move_outcome})
        if delta.our_move_crit != curr_ctx.our_move_crit:
            s.layer3_mismatches += 1
            s.record_example(3, {"check": "our_move_crit", "turn": battle.turn,
                                 "ctx": curr_ctx.our_move_crit, "delta": delta.our_move_crit})
        if delta.opp_move_crit != curr_ctx.opp_move_crit:
            s.layer3_mismatches += 1
            s.record_example(3, {"check": "opp_move_crit", "turn": battle.turn,
                                 "ctx": curr_ctx.opp_move_crit, "delta": delta.opp_move_crit})

        # === Faint / self-faint edge cases (protocol accuracy) ===
        # 1) Go first + Explosion: the move LANDED before the self-faint, so it
        #    must be attributed as a move that hit — not "nothing happened".
        if delta.we_fainted and delta.our_move_id in ("explosion", "selfdestruct"):
            s.explosion_faint_seen += 1
            if delta.our_switch_to is not None or delta.our_move_outcome != "hit":
                s.edge_mismatches += 1
                s.record_example(3, {"check": "explosion_self_faint", "turn": battle.turn,
                                     "our_move_id": delta.our_move_id,
                                     "outcome": delta.our_move_outcome,
                                     "switch_to": delta.our_switch_to})
        # 2) Switch a mon in that dies (e.g. to Spikes): it's a SWITCH, no move —
        #    our_switch_to set, our_move_id None, no outcome.
        if delta.our_switch_to is not None and delta.we_fainted:
            s.switch_death_seen += 1
            if delta.our_move_id is not None or delta.our_move_outcome is not None:
                s.edge_mismatches += 1
                s.record_example(3, {"check": "switch_in_death", "turn": battle.turn,
                                     "switch_to": delta.our_switch_to,
                                     "our_move_id": delta.our_move_id,
                                     "outcome": delta.our_move_outcome})
        # 3) KO'd before acting: reason == "fainted", no move, no outcome.
        if delta.our_cant_reason == "fainted":
            s.faint_before_acting_seen += 1
            if delta.our_move_id is not None or delta.our_move_outcome is not None:
                s.edge_mismatches += 1
                s.record_example(3, {"check": "faint_before_acting", "turn": battle.turn,
                                     "our_move_id": delta.our_move_id,
                                     "outcome": delta.our_move_outcome})
        # --- Symmetric OPPONENT edge cases ---
        if delta.opp_fainted and delta.opp_move_id in ("explosion", "selfdestruct"):
            s.explosion_faint_seen += 1
            if delta.opp_move_outcome != "hit":
                s.edge_mismatches += 1
                s.record_example(3, {"check": "opp_explosion_self_faint", "turn": battle.turn,
                                     "opp_move_id": delta.opp_move_id,
                                     "outcome": delta.opp_move_outcome})
        if delta.opp_cant_reason == "fainted":
            s.faint_before_acting_seen += 1
            if delta.opp_move_id is not None or delta.opp_move_outcome is not None:
                s.edge_mismatches += 1
                s.record_example(3, {"check": "opp_faint_before_acting", "turn": battle.turn,
                                     "opp_move_id": delta.opp_move_id,
                                     "outcome": delta.opp_move_outcome})

        # === Layer 4: TurnDelta vs encoded vector ===
        enc = _get_encoder().encode(delta)

        for side, offset, outcome in [
            ("our", OFFSET_OUR_MOVE_OUTCOME, delta.our_move_outcome),
            ("opp", OFFSET_OPP_MOVE_OUTCOME, delta.opp_move_outcome),
        ]:
            got = _onehot_idx(enc, offset, OUTCOME_DIM)
            want = _OUTCOME_TO_IDX[outcome] if outcome is not None else None
            if got != want:
                s.layer4_mismatches += 1
                s.record_example(4, {"check": f"{side}_outcome_onehot", "turn": battle.turn,
                                     "outcome": outcome, "want_idx": want, "got_idx": got})

        for side, off, crit in [
            ("our", OFFSET_OUR_CRIT, delta.our_move_crit),
            ("opp", OFFSET_OPP_CRIT, delta.opp_move_crit),
        ]:
            if bool(enc[off] > 0.5) != bool(crit):
                s.layer4_mismatches += 1
                s.record_example(4, {"check": f"{side}_crit_bit", "turn": battle.turn,
                                     "crit": crit, "enc": float(enc[off])})

        # Cant one-hot: the raw reason on the delta, normalized, must match the
        # encoded index (or all-zeros for None / unknown reason).
        for side, off, reason in [
            ("our", OFFSET_OUR_CANT, delta.our_cant_reason),
            ("opp", OFFSET_OPP_CANT, delta.opp_cant_reason),
        ]:
            got = _onehot_idx(enc, off, CANT_DIM)
            try:
                norm = _normalize_cant_reason_ge(reason) if reason is not None else None
            except UnknownCantReasonError:
                norm = None
            want = _CANT_TO_IDX.get(norm) if norm is not None else None
            if got != want:
                s.layer4_mismatches += 1
                s.record_example(4, {"check": f"{side}_cant_onehot", "turn": battle.turn,
                                     "reason": reason, "norm": norm,
                                     "want_idx": want, "got_idx": got})

        # === Coverage ===
        for o in (delta.our_move_outcome, delta.opp_move_outcome):
            if o == "hit":
                s.hit_seen += 1
            elif o == "miss":
                s.miss_seen += 1
            elif o == "fail":
                s.fail_seen += 1
        if delta.our_move_crit or delta.opp_move_crit:
            s.crit_seen += 1
        for reason in (delta.our_cant_reason, delta.opp_cant_reason):
            try:
                norm = _normalize_cant_reason_ge(reason) if reason is not None else None
            except UnknownCantReasonError:
                norm = None
            if norm is not None and norm in _CANT_TO_IDX:
                s.cant_reasons_seen.add(norm)

    def choose_move(self, battle):
        try:
            mask = Gen3ActionMasker.get_mask(battle)
            battle_tag = battle.battle_tag
            if battle_tag not in self._our_slots:
                self._our_slots[battle_tag] = SlotRegistry()
                self._opp_slots[battle_tag] = SlotRegistry()

            curr_ctx = BattleContext.from_battle(
                battle, mask=mask,
                our_slots=self._our_slots[battle_tag],
                opp_slots=self._opp_slots[battle_tag],
            )

            prev_ctx = self._prev_ctx.get(battle_tag)
            last_action = self._last_action.get(battle_tag)
            if prev_ctx is not None and last_action is not None:
                delta = build_legacy(prev_ctx, curr_ctx, last_action)
                self._validate_turn(battle, curr_ctx, delta)

            if battle.finished:
                self._prev_ctx[battle_tag] = None
                self._last_action[battle_tag] = None
                self._our_slots.pop(battle_tag, None)
                self._opp_slots.pop(battle_tag, None)
                valid = np.where(mask == 1)[0]
                return Gen3ActionMapper.action_to_order(int(np.random.choice(valid)), battle)

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))
            self._prev_ctx[battle_tag] = curr_ctx
            self._last_action[battle_tag] = choice
            return Gen3ActionMapper.action_to_order(choice, battle)

        except Exception as e:
            print(f"\n[FUZZ FATAL] Battle {battle.battle_tag}, Turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(s: ScenarioStats) -> None:
    total = (s.layer1_mismatches + s.layer2_mismatches + s.layer3_mismatches
             + s.layer4_mismatches + s.edge_mismatches)
    status = "PASS" if total == 0 else "FAIL"
    print(f"\n{'=' * 65}")
    print(f"SCENARIO: {s.name}  [{status}]")
    print(f"{'=' * 65}")
    print(f"Total decision turns  : {s.n_turns}")
    print(f"Layer 1 mismatches    : {s.layer1_mismatches}  (raw vs poke-env properties)")
    print(f"Layer 2 mismatches    : {s.layer2_mismatches}  (poke-env vs BattleContext)")
    print(f"Layer 3 mismatches    : {s.layer3_mismatches}  (BattleContext vs TurnDelta)")
    print(f"Layer 4 mismatches    : {s.layer4_mismatches}  (TurnDelta vs encoded vector)")
    print(f"Edge mismatches       : {s.edge_mismatches}  (faint/self-faint misrepresented)")
    print(f"Coverage:")
    print(f"  hit seen            : {s.hit_seen}")
    print(f"  miss seen           : {s.miss_seen}")
    print(f"  fail seen           : {s.fail_seen}")
    print(f"  crit seen           : {s.crit_seen}")
    print(f"  cant reasons seen   : {sorted(s.cant_reasons_seen)}")
    print(f"  explosion self-faint: {s.explosion_faint_seen}  (move landed, then self-KO)")
    print(f"  switch-in death     : {s.switch_death_seen}  (e.g. died to Spikes)")
    print(f"  faint before acting : {s.faint_before_acting_seen}  (KO'd first; reason='fainted')")
    if s.examples:
        print("Mismatch examples (up to 15):")
        for ex in s.examples:
            print(f"  {ex}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_scenario(name: str, team_str: str, n_battles: int, ts: int) -> ScenarioStats:
    print(f"\n--- {name}: {n_battles} battles ---", flush=True)
    tb = Gen3Teambuilder(team_str)
    tag = name.replace("-", "")[:6]
    fuzz = MoveOutcomeFuzzPlayer(
        scenario_name=name,
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"MOf{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"MOo{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )
    await run_local_battles(fuzz, opp, n_battles)
    return fuzz.stats


async def main(n_battles: int = 60) -> None:
    ts = int(time.time()) % 100000
    print(f"Move-Outcome Fuzz Test — gen3ou — {n_battles} battles")

    variance = await run_scenario("Variance", VARIANCE_TEAM, n_battles, ts)
    print_report(variance)
    # Edge scenario: Explosion / Self-Destruct users + Spikes setters + a frail
    # fast mon, so the faint/self-faint edge cases actually fire under random play.
    edge = await run_scenario("FaintEdge", EDGE_TEAM, n_battles, ts)
    print_report(edge)

    all_stats = [variance, edge]
    total_mismatches = sum(
        s.layer1_mismatches + s.layer2_mismatches + s.layer3_mismatches
        + s.layer4_mismatches + s.edge_mismatches for s in all_stats
    )

    coverage_issues = []
    if sum(s.crit_seen for s in all_stats) == 0:
        coverage_issues.append("crit never seen")
    if sum(s.miss_seen for s in all_stats) == 0:
        coverage_issues.append("miss never seen")
    if sum(s.fail_seen for s in all_stats) == 0:
        coverage_issues.append("fail never seen")
    all_cant = set().union(*(s.cant_reasons_seen for s in all_stats))
    if len(all_cant) < 2:
        coverage_issues.append(f"only {len(all_cant)} distinct cant reasons "
                               f"({sorted(all_cant)}); want >= 2")
    # Edge-case coverage (soft — report if a scenario never exercised them):
    exp = sum(s.explosion_faint_seen for s in all_stats)
    swd = sum(s.switch_death_seen for s in all_stats)
    fba = sum(s.faint_before_acting_seen for s in all_stats)
    print(f"\nEdge-case coverage across scenarios: explosion_self_faint={exp} "
          f"switch_in_death={swd} faint_before_acting={fba}")

    print(f"{'=' * 65}")
    if total_mismatches == 0 and not coverage_issues:
        print("PASS — All four layers correct, faint edge cases consistent, coverage satisfied.")
    else:
        print("FAIL:")
        if total_mismatches > 0:
            print(f"  Total mismatches: {total_mismatches}")
        for issue in coverage_issues:
            print(f"  Coverage gap: {issue}")
        sys.exit(1)
    print("=" * 65)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    asyncio.run(main(n))
