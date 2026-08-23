"""gen3_bait_entropy_v1 on REAL battles: the `bait_opportunity` flag's emission path through
`Gen3Env`, and the flag CROSS-CHECKED against the offline bait detector (`main.prober.loops`).

Why the cross-check exists. The detector is the bait definition of record — it is what will read the
probe's verdict — and it is a **behaviour** predicate (post hoc, from the raw protocol: "the opponent
voluntarily pivoted, we moved into it, and the move did nothing"). The env flag is an **opportunity**
predicate (at decision time, from public information: "the attack we would click is dead into a mon
sitting on their bench"). They are deliberately NOT the same set:

  * every detector `immune` whiff, on a turn where the arrival was already revealed, should have been
    a flagged decision — that direction IS asserted here, and it is the one that matters (a boost that
    misses the states the verdict is read on would probe nothing);
  * the reverse does NOT hold and is not asserted — a flagged decision where they do not pivot, or
    where we click something else, is an opportunity that never became a whiff. That gap is the point
    of an opportunity predicate: it fires BEFORE the mistake.

The one sanctioned disagreement is an ability immunity we had not yet seen (Levitate on an unrevealed
Flygon). That is the flag's documented public-information scope, so it is EXCUSED — but only after
checking that the ability really was unknown at that decision, never as a blanket allowance.

Marked `sim`: real battles in-process via the local BattleStream bridge — no server.
"""
from __future__ import annotations

import asyncio
import collections
import pathlib
from typing import Optional

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from agents.baitbot import Gen3BaitBotPlayer, blocks
from agents.battle.live_view import LegalActions
from agents.gen3_mechanics import effective_multiplier_by_types
from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env, _bait_candidate_attack
from main.prober import loops
from utils.bridge.bridge_session import attach_bridge_transport
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


# ----------------------------------------------------------------- 1. the emission path
def _candidate_action(battle, mask) -> Optional[int]:
    """The 11-way action index for `_bait_candidate_attack`, or None if it is not selectable.

    Driving the env with RANDOM legal actions does not work here and the failure is instructive:
    BaitBot only pivots into a mon immune to our REVEALED attacks, so a player that switches at
    random reveals nothing and manufactures no baits — measured 0 flag fires in 60 decisions on one
    seed, while the flag's own base rate is ~1% (the `--debug` smoke's `baitent/flagged_frac`). The
    firing assertion below therefore needs a player that actually commits to an attack.
    """
    active = battle.active_pokemon
    cand = _bait_candidate_attack(active, battle.available_moves or []) if active is not None else None
    if cand is None:
        return None
    legal = LegalActions.from_battle(battle)
    for slot, m in enumerate(legal.move_slots):     # action 6+k ⇔ request move slot k
        if m.id == cand.id and mask[6 + slot]:
            return 6 + slot
    return None


def test_gen3env_emits_the_bait_flag_over_the_bridge():
    """The obs key is declared, present on every decision, and binary — through the real
    `Gen3Env` → bridge path, against a BaitBot opponent that manufactures the boards."""
    teams = _teams()
    env = Gen3Env(
        load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration("BaitFlagEnv", None),
        start_listening=False, emit_bait_opportunity=True,
    )
    attach_bridge_transport(env, battle_format="gen3ou", persistent=True)
    opponent = Gen3BaitBotPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(teams), p_bait=1.0, seed=7,
        account_configuration=AccountConfiguration("BaitFlagOpp", None), start_listening=False,
    )
    wrapped = SingleAgentWrapper(env, opponent)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space

    assert "bait_opportunity" in env.observation_space.spaces
    rng = np.random.default_rng(0)
    seen, fired = 0, 0
    try:
        for _ in range(3):
            obs, _ = wrapped.reset()
            for _ in range(400):
                v = obs["bait_opportunity"]
                assert v.shape == (1,) and v.dtype == np.float32
                assert float(v[0]) in (0.0, 1.0)
                # The emitted column IS this decision's predicate — not a stale value, not the
                # neighbouring `defensive_opportunity` wired into the wrong key.
                assert float(v[0]) == Gen3Env._bait_opportunity(env)
                seen += 1
                fired += int(float(v[0]) == 1.0)
                mask = np.asarray(obs["action_mask"]).astype(bool)
                legal = np.flatnonzero(mask)
                act = _candidate_action(env.battle1, mask)
                if act is None:
                    act = int(rng.choice(legal)) if legal.size else 0
                obs, _r, term, trunc, _i = wrapped.step(act)
                if term or trunc:
                    break
    finally:
        with __import__("contextlib").suppress(Exception):
            wrapped.close()
    assert seen > 20, f"too few decisions to say anything (seen={seen}, fired={fired})"
    # NOTE — deliberately NO `fired > 0` assertion here, and the reason is a real property of the
    # predicate rather than a weak test. The flag needs the immune mon REVEALED and on the BENCH; a
    # p_bait=1.0 BaitBot pivots it IN and largely leaves it in, so the bench window is narrow, and the
    # base rate is ~1% anyway (the `--debug` smoke's `baitent/flagged_frac` 0.005-0.016). Over ~80
    # decisions a zero is ordinary — an assertion here would flap (measured: 4 failures in 10 runs).
    # The FIRING evidence is the cross-check below, which finds 21 flagged bait states per run.


# ----------------------------------------------------------------- 2. vs the offline detector
class _CandidateAttacker(Player):
    """Always clicks `_bait_candidate_attack` — the SHIPPED candidate rule — and records, per turn,
    what the env predicate saw.

    Clicking the candidate is what makes the comparison exact: the detector reports the move we
    ACTUALLY used, so any other policy would leave "we clicked a different move than the flag priced"
    as an untestable confound rather than a finding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lines_by_tag: dict[str, list[str]] = collections.defaultdict(list)
        self.dec_by_tag: dict[str, dict[int, dict]] = collections.defaultdict(dict)
        self.role_by_tag: dict[str, Optional[str]] = {}
        self._tag: Optional[str] = None

    async def _handle_battle_message(self, split_messages):
        await super()._handle_battle_message(split_messages)
        for m in split_messages:
            if m and m[0].startswith(">"):
                self._tag = m[0][1:]
                continue
            if self._tag is not None and m:
                self.lines_by_tag[self._tag].append("|".join(m))

    def choose_move(self, battle) -> BattleOrder:
        try:
            self._record(battle)
        except Exception:                       # a recording bug must not change the battles played
            pass
        active, moves = battle.active_pokemon, (battle.available_moves or [])
        cand = _bait_candidate_attack(active, moves) if active is not None else None
        if cand is not None:
            return Player.create_order(cand)
        return self.choose_random_move(battle)

    def _record(self, battle) -> None:
        self.role_by_tag[battle.battle_tag] = battle.player_role
        active, moves = battle.active_pokemon, (battle.available_moves or [])
        cand = _bait_candidate_attack(active, moves) if active is not None else None
        bench: dict[str, dict] = {}
        if cand is not None:
            for mon in (battle.opponent_team or {}).values():
                if mon is battle.opponent_active_pokemon or mon.active or mon.fainted:
                    continue
                sp = loops.norm_id(mon.species)
                bench[sp] = {
                    "blocked": bool(blocks(cand, mon)),
                    # Type chart ALONE, ability ignored — separates "we could have known" from
                    # "only a revealed ability would have told us".
                    "blocked_by_types": effective_multiplier_by_types(
                        cand.type, mon.type_1, mon.type_2, None) == 0.0,
                    "ability_known": bool(getattr(mon, "ability", None)),
                }
        self.dec_by_tag[battle.battle_tag][battle.turn] = {
            "flag": Gen3Env._bait_opportunity(type("_B", (), {"battle1": battle})()),
            "candidate": cand.id if cand is not None else None,
            "bench": bench,
        }


#: PINNED teams — the whole matchup is fixed (these two files + the sim seed below), because drawing
#: from the pool made this test a coin flip: over 14 random sample-team pairs only 2 produced any
#: immune whiff at all, and the pooled version scored `checked` anywhere from 0 to 48 run to run. A
#: cross-check whose sample size is a random variable cannot carry a floor.
_US_TEAM, _THEM_TEAM = "023a2d47648b85e6", "0972146213a667c9"


def _pinned_team(stem: str) -> str:
    p = pathlib.Path("data/teams/sample") / f"{stem}.txt"
    assert p.exists(), (f"pinned team {p} is gone — re-pin the pair (see the note above; the "
                        f"requirement is a matchup where our attacks meet an immunity)")
    return p.read_text()


def test_the_env_flag_fires_on_the_detectors_immune_whiffs():
    us = _CandidateAttacker(
        battle_format="gen3ou", team=Gen3Teambuilder([_pinned_team(_US_TEAM)]), max_concurrent_battles=1,
        account_configuration=AccountConfiguration("BaitXCheckUs", None), start_listening=False)
    them = Gen3BaitBotPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder([_pinned_team(_THEM_TEAM)]), p_bait=1.0, seed=11,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration("BaitXCheckOpp", None), start_listening=False)
    asyncio.run(run_local_battles(us, them, 4, battle_format="gen3ou", seed=[7, 11, 13, 17]))

    checked = excused = unrevealed = 0
    failures: list[str] = []
    for tag, lines in us.lines_by_tag.items():
        role = us.role_by_tag.get(tag)
        if role not in ("p1", "p2"):
            continue
        opp_side = "p2" if role == "p1" else "p1"
        decisions = us.dec_by_tag.get(tag, {})
        for b in loops.bait_events(loops.parse_events(lines), pivot_side=opp_side):
            if b.kind != "immune":
                continue                        # `fail` / `near_zero` are not immunity — out of scope
            dec = decisions.get(b.turn)
            if dec is None or dec["candidate"] is None:
                continue
            row = dec["bench"].get(b.arrival)
            if row is None:
                unrevealed += 1                 # arrival not a revealed, alive bench mon at decision time
                continue
            if not row["blocked"]:
                # The SIM says immune and the shipped predicate did not. The one sanctioned reason is
                # an ability we had not seen; anything else is a real disagreement with ground truth.
                if row["ability_known"] or row["blocked_by_types"]:
                    failures.append(
                        f"{tag} T{b.turn}: {dec['candidate']} vs {b.arrival} — sim says immune, "
                        f"predicate says no (ability_known={row['ability_known']}, "
                        f"by_types={row['blocked_by_types']})")
                else:
                    excused += 1
                continue
            checked += 1
            if dec["flag"] != 1.0:
                failures.append(f"{tag} T{b.turn}: {b.arrival} blocked our {dec['candidate']} on the "
                                f"bench, but bait_opportunity was {dec['flag']}")

    assert not failures, "\n".join(failures)
    # Non-vacuity: a run that found no immune whiffs proves nothing, and would pass silently.
    # MEASURED 2026-08-23 on the pinned matchup: 23 immune whiffs, 21 cross-checked, 2 arrivals still
    # unrevealed at the decision, 0 disagreements — stable across 6 repeats, so the floor has clearance.
    assert checked >= 8, (f"too few detector immune-whiffs were cross-checkable: checked={checked} "
                          f"(excused={excused}, unrevealed={unrevealed})")
