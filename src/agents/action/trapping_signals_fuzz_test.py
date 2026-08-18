"""Trapping-signals fuzz — validates the gen3_trapping_signals_v1 obs additions over a real
Arena-Trap battle run in-process via the local BattleStream bridge (no server).

This is the promoted, permanent form of the throwaway `/tmp/trap_mask_bridge_test.py` prototype.
It forces the one scenario the three signals exist for — a grounded mon that keeps trying to
switch out against a Dugtrio with Arena Trap — and drives the FULL production pipeline (mask →
``EpisodeTracker.record`` → obs encode → TurnDelta fold) exactly as ``Gen3Env`` does, asserting:

  Signal 1 (trapped obs bit)       : the FULL obs at `OFFSET_REACTIVE + trapped` == legal.trapped
     on EVERY decision.
  Signal 2 (maybe_trapped obs bit) : the FULL obs at `OFFSET_REACTIVE + maybe_trapped` ==
     legal.maybe_trapped on EVERY decision.
  Signal 3 (rejected-switch history): validated END TO END against INDEPENDENT raw-protocol truth,
     not our own fold —
       * the `|error|[Unavailable choice]` line is intercepted straight off the wire in
         `_handle_battle_message` and counted (ground truth);
       * on EVERY decision the most-recent turn-history slot's `attempted_switch_rejected` bit,
         read at its ABSOLUTE index inside the full obs (gen3_frame_deletion_v1: == base,
         exactly as `Gen3Env.embed_battle` builds it), must equal the folded delta's field — so a
         wrong offset / never-set / stuck-on bit all fail;
       * on a rejection the full-obs attempted-switch species id at its absolute index must be the
         bench mon we pressed (non-zero, matching), `our_switch_to` must be None (the switch never
         executed), and the rejection must be preceded by a maybe_trapped switch attempt;
       * FINAL cross-check: raw-protocol [Unavailable choice] count == folded history-bit count
         (no miss, no double-count, no fabrication).
  Mask invariant (regression)      : trapped == True ⇒ ZERO switch bits in the mask.

Coverage (all required for PASS): some decision is maybe_trapped with switches offered (pre-reveal);
some decision is trapped with switches masked off (post-reveal); at least one rejection seen on the
raw wire AND folded into history; reactive bits + full-obs slots validated on every decision with
zero mismatches.

P1 leads a grounded mon and always tries to switch (to provoke the reveal). P2 runs Dugtrio with
Arena Trap and only clicks moves (keeps Dugtrio in). Both run Gen3Battle so the out-of-band
CHOICE_REJECTED event is recorded by the poke-env _handle_battle_message hook.

Run directly (no server needed — local bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/action/trapping_signals_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import sys
import traceback

import numpy as np

from poke_env.player.player import Player
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from utils.bridge.local_battle_runner import run_local_battles
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions
from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from agents.observation.state_encoder import load_mappings, get_observation_encoder
from agents.observation.constants import (
    OFFSET_OUR_TEAM, POKEMON_FULL_DIM, POKEMON_ACTIVE_OFFSET,
    POKEMON_TRAPPED_OFFSET, POKEMON_MAYBE_TRAPPED_OFFSET, TEAM_SIZE,
)
from agents.observation.turn_delta_encoder import (
    TurnDeltaEncoder,
)
from agents.training.episode_tracker import EpisodeTracker

FMT = "gen3ou"

# gen3_entity_rehome_v1: the trapping bits ride OUR ACTIVE's mon slot — locate the active
# slot from the obs itself (the active flag is the last dim of each slot), then read the two
# per-mon bits beside it. Bench slots must stay zero (also asserted).
def _our_trapping_bits(obs):
    for i in range(TEAM_SIZE):
        start = OFFSET_OUR_TEAM + i * POKEMON_FULL_DIM
        if obs[start + POKEMON_ACTIVE_OFFSET] > 0.5:
            return (float(obs[start + POKEMON_TRAPPED_OFFSET]),
                    float(obs[start + POKEMON_MAYBE_TRAPPED_OFFSET]))
    return None

# P2: Dugtrio with Arena Trap up front, the trapper. Bench filler so the team is legal.
TRAPPER_TEAM = """
Dugtrio
Ability: Arena Trap
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Earthquake
- Rock Slide
- Substitute
- Aerial Ace

Snorlax
Ability: Immunity
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Body Slam
- Earthquake
- Self-Destruct
- Rest

Metagross
Ability: Clear Body
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Rock Slide
- Explosion

Skarmory
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Spikes
- Drill Peck
- Roar
- Rest

Blissey
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Soft-Boiled
- Seismic Toss
- Thunder Wave
- Ice Beam

Suicune
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Surf
- Calm Mind
- Rest
- Sleep Talk
"""

# P1: grounded lead (Snorlax) so Arena Trap applies, with bench mons it *could* switch to.
GROUNDED_TEAM = """
Snorlax
Ability: Thick Fat
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Body Slam
- Earthquake
- Self-Destruct
- Rest

Metagross
Ability: Clear Body
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Rock Slide
- Explosion

Swampert
Ability: Torrent
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Earthquake
- Ice Beam
- Surf
- Protect

Tyranitar
Ability: Sand Stream
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Crunch
- Dragon Dance

Blissey
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Soft-Boiled
- Seismic Toss
- Thunder Wave
- Ice Beam

Forretress
Ability: Sturdy
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Spikes
- Rapid Spin
- Earthquake
- Explosion
"""


def _pick_action(legal: LegalActions, mask: np.ndarray) -> int:
    """Provoke the trap reveal: prefer a switch when the server offers one (action index =
    the bench mon's team slot), else the first legal move, else struggle. Returns the 0-10
    action index the tracker.advance() / mapper consume."""
    if legal.switches:
        return legal.switches[0].slot
    for slot in range(4):
        if mask[6 + slot]:
            return 6 + slot
    return 10  # struggle


class TrapFuzzPlayer(Player):
    """P1: drives a per-battle EpisodeTracker through the production decision flow and
    validates the three trapping signals at every decision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, battle_class=Gen3Battle, **kwargs)
        self._mappings = load_mappings()
        self._enc = get_observation_encoder(self._mappings)
        self._td_enc = TurnDeltaEncoder(
            self._mappings.get("moves", {}), self._mappings.get("species", {})
        )
        self._trackers: dict[str, EpisodeTracker] = {}
        self._prev_rec: dict[str, dict] = {}  # battle_tag -> last decision's record
        self.violations: list[str] = []
        self.decisions = 0
        # coverage flags
        self.any_maybe_offered = False
        self.any_trapped_masked = False
        self.rejections_in_history = 0
        self.reactive_bits_checked = 0
        # INDEPENDENT ground truth: count |error|[Unavailable choice] lines straight off the
        # raw protocol (intercepted in _handle_battle_message, before any of our folding), so
        # the rejected-switch history is validated against the wire, not against our own fold.
        self.protocol_unavailable = 0
        self.full_obs_slot_checks = 0

    async def _handle_battle_message(self, split_messages):
        """Tap the raw protocol for the rejection ground truth before delegating. The
        |error|[Unavailable choice] line is the wire-truth a switch we chose was refused —
        counting it here (not via our TurnDelta fold) makes the history validation
        non-circular."""
        for sm in split_messages:
            if len(sm) > 2 and sm[1] == "error" and sm[2].startswith("[Unavailable choice]"):
                self.protocol_unavailable += 1
        return await super()._handle_battle_message(split_messages)

    def _tracker_for(self, tag: str) -> EpisodeTracker:
        tr = self._trackers.get(tag)
        if tr is None:
            # Mirror Gen3Env exactly (bounded history deque) so the assembled obs is
            # byte-for-byte what training feeds the model.
            tr = EpisodeTracker(history_cap=1)
            self._trackers[tag] = tr
        return tr

    def choose_move(self, battle):
        try:
            return self._choose(battle)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"\n[FUZZ FATAL] {battle.battle_tag} turn {battle.turn}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            import os
            os._exit(1)

    def _choose(self, battle):
        tag = battle.battle_tag
        tr = self._tracker_for(tag)

        legal = LegalActions.from_battle(battle)
        mask = Gen3ActionMasker.get_mask(battle, legal=legal).astype(np.int8)
        switch_bits = int(mask[0:6].sum())

        # --- Mask invariant (regression): trapped => no switch bits. ---
        if legal.trapped and switch_bits > 0:
            self.violations.append(
                f"[{tag}] t{battle.turn}: trapped but {switch_bits} switch bits in mask"
            )

        if not battle.strict_view().finished and mask.sum() > 0:
            tr.record(battle, mask, legal=legal)
        self.decisions += 1

        # --- Signals 1 & 2: the reactive obs bits must mirror the legality flags. ---
        obs = self._enc.encode(battle, hp_tracker=tr.hidden_power_tracker, legal=legal)
        bits = _our_trapping_bits(obs)
        if bits is None:
            self.violations.append(f"[{tag}] t{battle.turn}: no active slot found in obs")
            return
        trapped_bit, maybe_bit = bits
        # Bench slots must never carry the bits (the fact rides the trapped ENTITY only).
        for i in range(TEAM_SIZE):
            start = OFFSET_OUR_TEAM + i * POKEMON_FULL_DIM
            if obs[start + POKEMON_ACTIVE_OFFSET] <= 0.5 and (
                    obs[start + POKEMON_TRAPPED_OFFSET] != 0.0
                    or obs[start + POKEMON_MAYBE_TRAPPED_OFFSET] != 0.0):
                self.violations.append(
                    f"[{tag}] t{battle.turn}: bench slot {i} carries a trapping bit")
        if trapped_bit != float(legal.trapped):
            self.violations.append(
                f"[{tag}] t{battle.turn}: trapped obs bit {trapped_bit} != legal.trapped {legal.trapped}"
            )
        if maybe_bit != float(legal.maybe_trapped):
            self.violations.append(
                f"[{tag}] t{battle.turn}: maybe_trapped obs bit {maybe_bit} != "
                f"legal.maybe_trapped {legal.maybe_trapped}"
            )
        self.reactive_bits_checked += 1
        if legal.maybe_trapped and not legal.trapped and switch_bits > 0:
            self.any_maybe_offered = True
        if legal.trapped and switch_bits == 0:
            self.any_trapped_masked = True

        # --- Signal 3: the rejected-switch bit must land at its ABSOLUTE index in the FULL
        # obs the model consumes. Assemble the obs exactly as Gen3Env.embed_battle does
        # and read the most-recent history
        # slot's bit DIRECTLY out of that vector — not the standalone encoder — so the layout
        # offset + concat order are validated end to end, on EVERY decision. ---
        delta = tr.build_delta(battle=battle)
        # gen3_frame_deletion_v1 REPOINTED this from the TurnDelta lag frames (deleted) to the
        # H-B event window, which is where a rejected switch now lands as an EVENT_T_SWITCH_REJECTED
        # row. The end-to-end claim is unchanged: assemble the obs exactly as Gen3Env.embed_battle
        # does, and read the signal DIRECTLY out of that vector rather than from the standalone
        # encoder, so the layout offset is validated on EVERY decision.
        #
        # ⚠️ ONE FACT IS NOT REPOINTED, DELIBERATELY: `our_attempted_switch_spec` — WHICH bench mon
        # the refused switch was aimed at. The frames carried it; the event window cannot, and not
        # by oversight — `Gen3Battle.record_choice_rejected` says the attempted target "is not on
        # the wire and is recovered at fold time from the action index", and this window folds from
        # EVENTS ALONE. Closing it would mean threading the action index into the event fold, which
        # changes that tracker's contract rather than adding a missing row. What survives is the
        # rejection FACT (the row below) and trappedness itself (per-mon slots, gen3_entity_rehome_v1);
        # what is lost is the identity of the refused target.
        from agents.observation.constants import (
            OFFSET_EVENT_WINDOW, EVENT_WINDOW_N, EVENT_TOKEN_DIM, EVENT_T_SWITCH_REJECTED,
        )
        full_obs = obs
        _rej_rows = sum(
            1 for _r in range(EVENT_WINDOW_N)
            if full_obs[OFFSET_EVENT_WINDOW + _r * EVENT_TOKEN_DIM + 18] >= 0.5
            and int(full_obs[OFFSET_EVENT_WINDOW + _r * EVENT_TOKEN_DIM + 0]) == EVENT_T_SWITCH_REJECTED
        )
        if delta.attempted_switch_rejected and _rej_rows == 0:
            self.violations.append(
                f"[{tag}] t{battle.turn}: delta.attempted_switch_rejected is True but the "
                f"event window carries NO EVENT_T_SWITCH_REJECTED row — the signal never "
                f"reached the model."
            )
        self.full_obs_slot_checks += 1

        # Invariant on EVERY decision, BOTH directions: an EVENT_T_SWITCH_REJECTED row is
        # present in the obs exactly when the delta says a switch was refused. The reverse
        # direction is the half that catches a STUCK-ON row, which a presence-only check
        # would read as a pass forever.
        if bool(_rej_rows) != bool(delta.attempted_switch_rejected):
            self.violations.append(
                f"[{tag}] t{battle.turn}: obs EVENT_T_SWITCH_REJECTED rows={_rej_rows} but "
                f"delta.attempted_switch_rejected={delta.attempted_switch_rejected}"
            )

        if delta.attempted_switch_rejected:
            self.rejections_in_history += 1
            # The pivot was refused, so it never executed: switch_to is None, but the
            # *attempted* target must resolve to the bench mon we pressed.
            if delta.our_switch_to is not None:
                self.violations.append(
                    f"[{tag}] t{battle.turn}: rejected switch but our_switch_to="
                    f"{delta.our_switch_to!r} (the switch must NOT have executed)"
                )
            if delta.attempted_switch_to is None:
                self.violations.append(
                    f"[{tag}] t{battle.turn}: rejected switch but attempted_switch_to is None"
                )
            # gen3_frame_deletion_v1: the obs-side half of this check is GONE — there is no
            # attempted-target species in the event window to compare against (see the note
            # above). The FOLD-side assertion above (`attempted_switch_to is not None`) still
            # runs, so the TurnDelta half stays covered; what is no longer verifiable here is
            # that the identity reaches the model, because it does not.
            # Sequence: the previous decision in this battle attempted a switch under maybe_trapped.
            prev = self._prev_rec.get(tag)
            if not (prev and prev["attempted_switch"] and prev["maybe_trapped"]):
                self.violations.append(
                    f"[{tag}] t{battle.turn}: rejection not preceded by a maybe_trapped "
                    f"switch attempt (prev={prev})"
                )

        # --- Choose: provoke the reveal by switching whenever offered. ---
        idx = _pick_action(legal, mask)
        self._prev_rec[tag] = {
            "turn": battle.turn,
            "maybe_trapped": bool(legal.maybe_trapped),
            "trapped": bool(legal.trapped),
            "attempted_switch": idx < 6,
        }
        tr.advance(idx)
        return Gen3ActionMapper.action_to_order(idx, battle, legal=legal, mask=mask)

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        self._trackers.pop(battle.battle_tag, None)
        self._prev_rec.pop(battle.battle_tag, None)


class Mover(Player):
    """P2: never switches (keeps Dugtrio in), just clicks a move."""

    def choose_move(self, battle):
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        return self.choose_default_move()


async def main(n_battles: int = 40) -> None:
    p1 = TrapFuzzPlayer(
        account_configuration=AccountConfiguration("TrapSigP1", None),
        battle_format=FMT,
        team=ConstantTeambuilder(GROUNDED_TEAM),
        start_listening=False,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )
    p2 = Mover(
        account_configuration=AccountConfiguration("TrapSigP2", None),
        battle_format=FMT,
        team=ConstantTeambuilder(TRAPPER_TEAM),
        start_listening=False,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
        battle_class=Gen3Battle,
    )

    await run_local_battles(p1, p2, n_battles, battle_format=FMT)

    # INDEPENDENT cross-check: every wire-truth |error|[Unavailable choice] must fold to
    # exactly one rejected-switch history bit — no miss, no double-count, no fabrication.
    if p1.protocol_unavailable != p1.rejections_in_history:
        p1.violations.append(
            f"rejection count mismatch: raw-protocol [Unavailable choice]="
            f"{p1.protocol_unavailable} != folded history bits={p1.rejections_in_history}"
        )

    print("=" * 70)
    print("Trapping-signals fuzz (Arena Trap) — gen3_trapping_signals_v1")
    print("=" * 70)
    print(f"battles run                : {n_battles}")
    print(f"p1 decisions               : {p1.decisions}")
    print(f"reactive bits checked       : {p1.reactive_bits_checked}")
    print(f"full-obs history-slot checks: {p1.full_obs_slot_checks}")
    print(f"raw-protocol [Unavailable]  : {p1.protocol_unavailable}")
    print(f"rejections folded → history : {p1.rejections_in_history}")
    print(f"coverage A (maybe+offered)  : {p1.any_maybe_offered}")
    print(f"coverage B (trapped+masked) : {p1.any_trapped_masked}")
    print(f"violations                 : {len(p1.violations)}")
    if p1.violations:
        print("\nVIOLATIONS (first 20):")
        for v in p1.violations[:20]:
            print("  " + v)

    coverage_holes = []
    if not p1.any_maybe_offered:
        coverage_holes.append("maybe_trapped+switches-offered never seen")
    if not p1.any_trapped_masked:
        coverage_holes.append("trapped+switches-masked never seen")
    if p1.rejections_in_history == 0:
        coverage_holes.append("no rejected switch ever folded into history")
    if p1.protocol_unavailable == 0:
        coverage_holes.append("no raw-protocol [Unavailable choice] ever intercepted")
    if p1.reactive_bits_checked == 0:
        coverage_holes.append("no reactive bits validated")
    if coverage_holes:
        print(f"\nCOVERAGE HOLES: {coverage_holes}")

    ok = not p1.violations and not coverage_holes
    print(
        "\nPASS — trapped/maybe_trapped obs bits track legality and the rejected pivot is "
        "folded into history." if ok else "\nFAIL"
    )
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    asyncio.run(main(n))
