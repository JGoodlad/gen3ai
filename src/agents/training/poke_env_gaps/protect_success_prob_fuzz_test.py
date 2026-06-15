"""Bridge fuzz test for the Protect-success-odds obs feature (gen3_protect_odds_v1).

Two things are validated against REAL in-process bridge battles (no server), and a violation
of either raises immediately:

  A. OBS CORRECTNESS (exact, per decision). The two reactive scalars — vec[OFFSET_REACTIVE+15]
     (our active) and vec[OFFSET_REACTIVE+16] (opp active) — MUST equal
     ``gen3_mechanics.protect_success_probability(mon.protect_counter)`` for the live mon, read
     through the SAME LiveView the encoder uses. This pins the wiring (poke-env counter →
     LiveView → encoder → correct offset → correct floored-doubling formula) and the offsets.

  B. EMPIRICAL % (the "are the probabilities actually right" check). One side spams Protect and
     the opponent always attacks; we intercept the raw |...| protocol to reconstruct, per protect
     attempt, the consecutive-stall index ``k`` and whether it SUCCEEDED, then assert the measured
     P(success | k) matches the SHIPPED formula ``1 / min(2**k, 8)`` within a binomial tolerance:
     100% / 50% / 25% / 12.5%-floor — NOT the base-condition *3 (100/33/11) and NOT unbounded
     halving. Plus the three RESET cases (the protect right AFTER a fail / a non-stall move / a
     switch must be back to 100%): "protect, fail, protect" → 100%, the user's worked example.

Gen3 mechanic (verified against the compiled sim): Showdown's gen3 stall condition inherits
gen4→gen5, NOT the base data/conditions.ts. The counter starts at 2 and DOUBLES each consecutive
successful stall move (gen5), capped at 8 (gen4 ``counterMax`` → "does not fall below 1/8").

Run directly (no server needed; in-process via the local BattleStream bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/protect_success_prob_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from poke_env import AccountConfiguration
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.gen3_mechanics import protect_success_probability
from agents.observation.constants import OFFSET_REACTIVE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

BATTLE_FORMAT = "gen3ou"
PROTECT_IDS = {"protect", "detect"}   # gen3 stall moves a spammer can repeat (Endure shares the counter)
_TOL = 1e-6

# Focal = a bulky all-Protect wall stack (Protect on every mon → chains every battle); opponent =
# pure Choice-Band attackers so willAct() always holds and the chain is pressured turn after turn.
FOCAL_TEAM = """\
Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Protect
- Spikes
- Drill Peck
- Roar

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Protect
- Soft-Boiled
- Seismic Toss
- Toxic

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Protect
- Surf
- Rest
- Calm Mind

Forretress @ Leftovers
Ability: Sturdy
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Protect
- Spikes
- Rapid Spin
- Earthquake

Swampert @ Leftovers
Ability: Torrent
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Protect
- Surf
- Earthquake
- Ice Beam

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Protect
- Shadow Ball
- Thunderbolt
- Ice Punch
"""

OPP_TEAM = """\
Tyranitar @ Choice Band
Ability: Sand Stream
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Rock Slide
- Earthquake
- Crunch
- Body Slam

Metagross @ Choice Band
Ability: Clear Body
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Meteor Mash
- Earthquake
- Rock Slide
- Body Slam

Salamence @ Choice Band
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Body Slam
- Brick Break

Aerodactyl @ Choice Band
Ability: Rock Head
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Double-Edge
- Hidden Power Flying

Heracross @ Choice Band
Ability: Guts
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Brick Break
- Megahorn

Donphan @ Leftovers
Ability: Sturdy
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Earthquake
- Rock Slide
- Body Slam
- Hidden Power Bug
"""


@dataclass
class _Stats:
    # --- Part A: obs correctness ---
    decisions: int = 0
    obs_fail: int = 0
    max_counter_seen: int = 0              # the highest protect_counter the obs was validated at
    counter_hist: dict = field(default_factory=lambda: defaultdict(int))  # k -> #decisions
    # --- Part B: empirical P(success | k) ---
    by_k: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # k -> [succ, att]
    after_fail: list = field(default_factory=lambda: [0, 0])          # protect right after a FAIL
    after_other_move: list = field(default_factory=lambda: [0, 0])    # ... after a non-stall move
    after_switch: list = field(default_factory=lambda: [0, 0])        # ... after a switch out+back
    examples: List[dict] = field(default_factory=list)

    def record(self, ex: dict) -> None:
        if len(self.examples) < 12:
            self.examples.append(ex)


class _ProtectFuzzPlayer(Player):
    """Spams Protect (else strongest attack; never switches) so chains run long; validates the
    obs scalars every decision and reconstructs P(success|k) from the protocol stream."""

    def __init__(self, *args, stats: _Stats, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)   # need live_view() / strict_view
        super().__init__(*args, **kwargs)
        self.stats = stats
        self.encoder = Gen3ObservationEncoder(load_mappings())
        layout = self.encoder.reactive_encoder.get_layout()
        self._our_off = OFFSET_REACTIVE + layout["protect_odds_our"]["offset"]
        self._opp_off = OFFSET_REACTIVE + layout["protect_odds_opp"]["offset"]
        # protocol-tracking state (per battle tag)
        self._consec = defaultdict(lambda: defaultdict(int))   # tag -> mon -> consecutive successes
        self._pending = {}                                     # tag -> (mon, k) | None
        self._pending_after = {}                               # tag -> reset-type of the pending protect
        self._next_after = defaultdict(lambda: defaultdict(lambda: None))  # tag -> mon -> reset for NEXT protect

    # ---------------------------------------------------------------- Part A
    def _validate_obs(self, battle) -> None:
        if battle.force_switch:        # not a normal move decision; active may be the fainted mon
            return
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if active is None:
            return
        s = self.stats
        s.decisions += 1
        vec = self.encoder.encode(battle)

        # The encoder reads protect_counter through the LiveView; mirror its None-guard exactly
        # (opp_active None → the encoder leaves 0.0, NOT protect_success_probability(0)=1.0).
        exp_our = protect_success_probability(active.protect_counter)
        exp_opp = protect_success_probability(opp.protect_counter) if opp is not None else 0.0
        got_our = float(vec[self._our_off])
        got_opp = float(vec[self._opp_off])

        s.counter_hist[int(active.protect_counter)] += 1
        s.max_counter_seen = max(s.max_counter_seen, int(active.protect_counter))

        if abs(got_our - exp_our) > _TOL or abs(got_opp - exp_opp) > _TOL:
            s.obs_fail += 1
            s.record({"check": "obs_scalar_mismatch", "turn": battle.turn,
                      "our_counter": int(active.protect_counter), "our_got": got_our, "our_exp": exp_our,
                      "opp_counter": (int(opp.protect_counter) if opp is not None else None),
                      "opp_got": got_opp, "opp_exp": exp_opp})

    # ---------------------------------------------------------------- Part B (protocol)
    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">")
        b = self._battles.get(tag)
        role = b.player_role if b is not None else None
        consec = self._consec[tag]
        if tag not in self._pending:
            self._pending[tag] = None
            self._pending_after[tag] = None
        next_after = self._next_after[tag]

        for msg in split_messages[1:]:
            if len(msg) < 2:
                continue
            mt = msg[1]
            if mt in ("switch", "drag") and len(msg) >= 3 and role is not None and msg[2][:2] == role:
                mon = msg[2].split(":", 1)[-1].strip()
                consec[mon] = 0
                next_after[mon] = "switch"
            elif mt == "move" and len(msg) >= 4 and role is not None and msg[2][:2] == role:
                mon = msg[2].split(":", 1)[-1].strip()
                name_id = msg[3].replace(" ", "").replace("-", "").lower()
                if name_id in PROTECT_IDS:
                    # k == the live protect_counter (successes already landed, 0-indexed) the attempt
                    # faces — so the expected odds are protect_success_probability(k), the SAME formula
                    # and 0-indexing the obs uses. k=0 is the first/post-reset protect (100%).
                    self._pending[tag] = (mon, consec[mon])
                    self._pending_after[tag] = next_after[mon]
                    next_after[mon] = None
                else:                                                   # non-protect move breaks the chain
                    if consec[mon] > 0:
                        next_after[mon] = "other_move"
                    consec[mon] = 0
            elif mt == "-singleturn" and len(msg) >= 4 and role is not None and msg[2][:2] == role \
                    and "Protect" in msg[3] and self._pending[tag] is not None:
                self._resolve(tag, success=True, consec=consec)
            elif mt == "-fail" and len(msg) >= 3 and role is not None and msg[2][:2] == role \
                    and self._pending[tag] is not None:
                self._resolve(tag, success=False, consec=consec, next_after=next_after)
            elif mt == "turn" and self._pending[tag] is not None:
                self._resolve(tag, success=False, consec=consec, next_after=next_after)  # no -singleturn ⇒ fail

        await super()._handle_battle_message(split_messages)

    def _resolve(self, tag, *, success, consec, next_after=None):
        mon, k = self._pending[tag]
        after = self._pending_after[tag]
        s = self.stats
        s.by_k[k][1] += 1
        s.by_k[k][0] += int(success)
        if after == "fail":
            s.after_fail[1] += 1; s.after_fail[0] += int(success)
        elif after == "other_move":
            s.after_other_move[1] += 1; s.after_other_move[0] += int(success)
        elif after == "switch":
            s.after_switch[1] += 1; s.after_switch[0] += int(success)
        if success:
            consec[mon] += 1
        else:
            consec[mon] = 0
            if next_after is not None:
                next_after[mon] = "fail"
        self._pending[tag] = None
        self._pending_after[tag] = None

    # ---------------------------------------------------------------- policy
    def _pick_order(self, battle):
        for mv in battle.available_moves:
            if mv.id in PROTECT_IDS:
                return self.create_order(mv)
        dmg = [m for m in battle.available_moves if m.base_power > 0]
        if dmg:
            return self.create_order(max(dmg, key=lambda m: m.base_power))
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:                  # forced switch (fainted) → chain ends
            return self.create_order(battle.available_switches[0])
        return self.choose_default_move()

    def choose_move(self, battle):
        try:
            self._validate_obs(battle)
            return self._pick_order(battle)
        except Exception as e:  # noqa: BLE001 — surface the exact decision that broke
            print("\n🛑 [PROTECT-ODDS FUZZ CRITICAL FAILURE] 🛑")
            print(f"Battle {battle.battle_tag} turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)


class _ResetCyclePlayer(_ProtectFuzzPlayer):
    """Deliberately drives the two reset patterns the pure spammer rarely makes — protect→non-stall
    move→protect (after_other_move) and protect→switch-out→switch-back→protect (after_switch) — via
    a per-battle phase cycle, while reusing all the obs validation + protocol bookkeeping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase = defaultdict(int)

    def _pick_order(self, battle):
        tag = battle.battle_tag
        avail = {m.id: m for m in battle.available_moves}

        def protect():
            for mid in PROTECT_IDS:
                if mid in avail:
                    return self.create_order(avail[mid])
            return None

        def strongest():
            dmg = [m for m in battle.available_moves if m.base_power > 0]
            if dmg:
                return self.create_order(max(dmg, key=lambda m: m.base_power))
            return self.create_order(battle.available_moves[0]) if battle.available_moves else None

        if battle.available_moves:
            ph = self._phase[tag]
            # 0 protect → 1 non-stall move → 2 protect(after_other_move) → 3 switch-out
            #           → 4 switch-back → 5 protect(after_switch) → 0
            if ph == 0 and (o := protect()) is not None:
                self._phase[tag] = 1; return o
            if ph == 1 and (o := strongest()) is not None:
                self._phase[tag] = 2; return o
            if ph == 2 and (o := protect()) is not None:
                self._phase[tag] = 3; return o
            if ph == 3 and battle.available_switches:
                self._phase[tag] = 4; return self.create_order(battle.available_switches[0])
            if ph == 4 and battle.available_switches:
                self._phase[tag] = 5; return self.create_order(battle.available_switches[0])
            if ph == 5 and (o := protect()) is not None:
                self._phase[tag] = 0; return o
            o = protect() or strongest()
            if o is not None:
                return o
        if battle.available_switches:
            self._phase[tag] = 0
            return self.create_order(battle.available_switches[0])
        return self.choose_default_move()


def _tol(p: float, n: int) -> float:
    """A generous binomial band: 0.04 + 3·SE. Wide enough to never flake, tight enough that the
    base-condition *3 rule (0.333 at k=2) or unbounded halving fails the assertion."""
    return 0.04 + 3.0 * math.sqrt(max(p * (1.0 - p), 1e-4) / n) if n > 0 else 1.0


async def _run_phase(name, cls, n_battles, ts, stats) -> None:
    focal = cls(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(FOCAL_TEAM),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"PoF{ts}{name[:4]}", "x"),
        max_concurrent_battles=10, stats=stats)
    opp = _AttackerPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(OPP_TEAM),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"PoO{ts}{name[:4]}", "x"),
        max_concurrent_battles=10)
    print(f"--- phase {name}: {n_battles} battles ---", flush=True)
    await run_local_battles(focal, opp, n_battles, concurrency=8)


class _AttackerPlayer(Player):
    def choose_move(self, battle):
        dmg = [m for m in battle.available_moves if m.base_power > 0]
        if dmg:
            return self.create_order(max(dmg, key=lambda m: m.base_power))
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_default_move()


async def run(n_battles: int) -> None:
    print(f"Protect-success-odds Fuzz — gen3ou bridge (no server) — {n_battles} battles\n")
    ts = int(time.time()) % 100000
    stats = _Stats()
    await _run_phase("spam", _ProtectFuzzPlayer, n_battles, ts, stats)
    await _run_phase("reset", _ResetCyclePlayer, max(60, n_battles // 2), ts + 1, stats)

    s = stats
    print("=" * 70)
    print(f"decisions validated (obs)  : {s.decisions}")
    print(f"obs scalar mismatches      : {s.obs_fail}")
    print(f"max protect_counter in obs : {s.max_counter_seen}")
    print(f"counter histogram (k:#dec) : {dict(sorted(s.counter_hist.items()))}")
    print("\nMEASURED P(protect success | consecutive-stall k)  vs the shipped formula")
    print(f"{'k':>3} {'attempts':>9} {'success':>8} {'rate':>7}  {'formula':>7}  {'tol':>6}")
    for k in sorted(s.by_k):
        succ, att = s.by_k[k]
        if att:
            exp = protect_success_probability(k)
            print(f"{k:>3} {att:>9} {succ:>8} {succ/att:>7.3f}  {exp:>7.3f}  {_tol(exp, att):>6.3f}")

    def fmt(b):
        return f"{b[0]}/{b[1]} = {b[0]/b[1]:.3f}" if b[1] else f"{b[0]}/{b[1]} = n/a"
    print("\nRESET CASES (the protect right AFTER the reset event → must be 100%):")
    print(f"  after protect FAIL    : {fmt(s.after_fail)}")
    print(f"  after non-stall move  : {fmt(s.after_other_move)}")
    print(f"  after switch out+back : {fmt(s.after_switch)}")
    for ex in s.examples:
        print("   example:", ex)
    print("=" * 70)

    # ---- assertions ----
    failures: List[str] = []
    inconclusive: List[str] = []

    # (A) obs correctness — exact, zero tolerance
    if s.obs_fail:
        failures.append(f"{s.obs_fail} obs scalar mismatch(es) vs protect_success_probability(counter)")
    if s.max_counter_seen < 2:
        inconclusive.append("obs never validated at protect_counter ≥ 2 — chains too short; run more battles")

    # (B) empirical % vs the shipped formula (only where the sample is large enough to be meaningful)
    _MIN_N = 40
    checked_ks = []
    for k in (0, 1, 2):     # counter 0/1/2 → 1.0 / 0.5 / 0.25 (counter≥3 is the floor, pooled below)
        succ, att = s.by_k.get(k, [0, 0])
        if att < _MIN_N:
            inconclusive.append(f"k={k}: only {att} attempts (<{_MIN_N}) — not asserted")
            continue
        exp, rate, tol = protect_success_probability(k), succ / att, _tol(protect_success_probability(k), att)
        checked_ks.append(k)
        if abs(rate - exp) > tol:
            failures.append(f"k={k}: empirical {rate:.3f} vs formula {exp:.3f} (tol {tol:.3f}, n={att})")
    # the 12.5% FLOOR: pool counter≥3 (each thin alone) and check it sits at 0.125, not unbounded 0.0625
    f_succ = sum(s.by_k[k][0] for k in s.by_k if k >= 3)
    f_att = sum(s.by_k[k][1] for k in s.by_k if k >= 3)
    if f_att >= _MIN_N:
        rate = f_succ / f_att
        if abs(rate - 0.125) > _tol(0.125, f_att):
            failures.append(f"floor (k≥3): empirical {rate:.3f} vs 0.125 (n={f_att})")
        else:
            checked_ks.append("≥3")
    else:
        inconclusive.append(f"floor (k≥3): only {f_att} attempts — floor not asserted")

    # reset cases — every protect after a reset must succeed (100%)
    for label, bucket in (("after_fail", s.after_fail), ("after_other_move", s.after_other_move),
                          ("after_switch", s.after_switch)):
        succ, att = bucket
        if att >= 20:
            if succ / att < 0.99:
                failures.append(f"reset {label}: {succ}/{att} = {succ/att:.3f} succeeded (expected ~1.0)")
        else:
            inconclusive.append(f"reset {label}: only {att} samples — not asserted")

    for w in inconclusive:
        print(f"⚠️  inconclusive: {w}")
    if failures:
        print("\n❌ FAIL")
        for f in failures:
            print("   -", f)
        os._exit(1)
    print(f"\n✅ PASS — obs exact over {s.decisions} decisions (counter≤{s.max_counter_seen}); "
          f"empirical P(success|k) matched the formula for k∈{checked_ks} + reset cases.")


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    asyncio.run(run(num))
