"""Per-field BIT-IDENTITY gate for the reward manager's two fast paths.

Both are changes to THE OBJECTIVE — a wrong reward trains a wrong policy with no error anywhere,
which is strictly worse than a slow one — so both are gated on the strongest evidence available:
real gen3ou battles through the local BattleStream bridge (no server), and at EVERY decision,
EVERY field of the `RewardBreakdown` compared against a full-computation twin.

  * **The suppressed-term skip** (`gen3_reward_skip_suppressed_v1`). `process_turn_reward` no
    longer computes the ~20 BIAS helpers whose results this run's composition forces to zero.
  * **The belief-block memo** (`gen3_belief_block_memo_v1`). Φ_belief's `encode_block` — 60.0% of
    `process_turn_reward`, the largest single item in the per-decision CPU budget — now answers
    from a CONTENT-keyed cache (`incoming_damage_encoder.IncomingBeliefMemo`).

ONE twin gates both, because `_shadow=True` means "compute everything the slow way": the twin runs
with the skip disabled AND with no memo at all. So a divergence in either mechanism shows up in
the same per-field comparison, and neither can hide behind the other by making the same mistake
twice.

Five things are checked per decision, per composition:

  1. **Bit-identity.** The production manager (fast paths ON) and a `_shadow=True` twin (skip OFF,
     memo OFF), driven in lockstep through the identical `record_action` / `process_turn_reward`
     sequence, must agree on every declared field and on `total` — compared with `!=`, not
     `isclose`. A fast path that moves a reward by 1e-16 is still one that changed the objective.
  2. **The skip is exactly the suppression's COMPLEMENT.** Every BIAS field the composition
     reports inactive must be 0.0 on the FULL path too. If it isn't, the skip is dropping a
     term the run still charges — the one way this change can be wrong.
  3. **Finiteness**, so a NaN can't make (1) vacuously true (NaN != NaN would fail, but an inf
     on both sides would pass — this catches it).
  4. **The memo's KEY is complete, differentially and on real boards.** Every decision's
     `attacker_state_key(live)` is recorded against the `AttackerThreat` freshly derived from that
     board; a key seen twice must carry the identical threat. This is the memo's central claim
     ("equal key ⇒ equal inputs") tested against thousands of real boards rather than a synthetic
     matrix — an under-key shows up here as a same-key/different-threat pair, with the offending
     field named. The structural counterpart (an AST walk over `_attacker_threat`) is
     `agents/observation/incoming_damage_memo_test.py`.
  5. **The memo actually SERVED.** Hit counts are reported and floored: a clean run in which the
     cache never returned anything would be evidence about the uncached path only.

Three compositions are swept **on the same decision stream** (one player, one set of battles,
three manager pairs), because the skip must be a no-op exactly where nothing is suppressed:

  * production default            — 1 BIAS term active; ~20 skipped
  * `--no-all-shaping-pbrs`       — the full additive BIAS class; almost nothing skipped
  * `--stall-pbrs` (+ all-shaping) — the WHOLE BIAS class zeroed; everything skipped

Sharing the decision stream is what makes the trigger-coverage report meaningful: the
`--no-all-shaping-pbrs` arm's non-zero counts are, turn for turn, the signals the production
arm SKIPPED. A run reporting "0 violations" alongside a coverage table of zeros would be
telling you nothing, so the table is printed and a floor is enforced on it.

Run directly (no server needed — runs via the local bridge):
    python src/agents/training/reward_skip_parity_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import asyncio
import math
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import fields as dc_fields
from typing import Optional

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions
from agents.observation.incoming_damage_encoder import (
    _attacker_threat as _fresh_attacker_threat,
    attacker_state_key,
)
from agents.training.battle_snapshot import BattleContext
from agents.training.progress_clock import ProgressClock
from agents.training.reward_manager import (
    Gen3RewardManager,
    RewardBreakdown,
    RewardClass,
    RewardConfig,
    reward_class_composition,
)
from agents.training.slot_registry import SlotRegistry
from agents.training.turn_delta import TurnDelta
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# The three documented compositions. Keys are the CLI spelling a reader would recognise.
COMPOSITIONS = {
    "default": RewardConfig(),
    "--no-all-shaping-pbrs": RewardConfig(all_shaping_pbrs=False),
    "--stall-pbrs": RewardConfig(all_shaping_pbrs=True, stall_pbrs=True),
}

# The arm whose non-zero BIAS counts stand in as the trigger coverage for what the other arms
# skipped (same decisions, same battles — see the module docstring).
_COVERAGE_ARM = "--no-all-shaping-pbrs"

# A "0 violations" run is only trustworthy if the corpus actually FIRED the skipped signals.
# These are the BIAS terms random play reaches often enough to demand; the rest (explosion_block,
# sleep_*, struggle_tax, …) need specific board states and are reported without a floor.
_REQUIRED_COVERAGE = ("status", "spikes", "futile_attack", "se_switch", "switch_base",
                      "matchup_penalty", "finishing_blow", "pivot_damage")

_BIAS_FIELDS = RewardBreakdown.registry_fields(RewardClass.BIAS)


class _ArmState:
    """One composition's manager pair for one battle: the production path and its oracle."""

    def __init__(self, config: RewardConfig, clock: ProgressClock):
        self.fast = Gen3RewardManager(config=config, progress_clock=clock)   # skip ON
        self.full = Gen3RewardManager(config=config, progress_clock=clock,   # skip OFF
                                      _shadow=True)


class _BattleState:
    def __init__(self):
        # ONE clock per battle, shared by every arm and both managers in each — faithful to
        # production, where the clock is owned by EpisodeTracker and only READ by the reward.
        # Without it `_apply_progress_clock` early-returns and `no_progress_tax` — the ONE term
        # the production composition still charges — would be structurally 0 all run, so the
        # fuzz would never exercise the one thing the fast path must not skip.
        self.clock = ProgressClock()
        self.arms = {name: _ArmState(cfg, self.clock) for name, cfg in COMPOSITIONS.items()}
        self.prev_ctx: Optional[BattleContext] = None
        self.last_action: Optional[int] = None
        self.prev_cursor: int = 0
        self.our_slots = SlotRegistry()
        self.opp_slots = SlotRegistry()


class RewardSkipParityPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turns_compared = 0
        self.violations: list[str] = []
        # per-arm, per-field count of non-zero FINAL values
        self.activations: dict[str, Counter] = {n: Counter() for n in COMPOSITIONS}
        self._per_battle: dict[str, _BattleState] = {}
        # (4) the differential key-coverage corpus: attacker key → the threat it must always mean.
        # Global across battles ON PURPOSE — the key is CONTENT, so two different battles reaching
        # the same board content must reach the same belief, and a cross-battle collision is the
        # sharpest form of the under-key bug (it is also exactly what a shared cache would serve).
        self._key_corpus: dict = {}
        self.keys_seen = 0
        self.key_repeats = 0
        self.memo_hits = 0
        self.memo_misses = 0

    def _get_state(self, battle) -> _BattleState:
        tag = battle.battle_tag
        if tag not in self._per_battle:
            self._per_battle[tag] = _BattleState()
        return self._per_battle[tag]

    def _fail(self, msg: str) -> None:
        self.violations.append(msg)
        print(f"\n[REWARD SKIP PARITY] {msg}", flush=True)
        traceback.print_stack()
        os._exit(1)

    def _check_attacker_key(self, battle, where: str) -> None:
        """(4) The memo's central claim, tested differentially on a real board.

        `attacker_state_key(live)` is asserted to DETERMINE `_attacker_threat(live, None)`. An
        under-key — an input the belief reads that the key does not carry — surfaces here as two
        boards with the same key and different beliefs, and the failure names the field, which is
        the diagnosis. Nothing is cached for the manager here; this recomputes both sides fresh.
        """
        try:
            live = battle.strict_view().live
        except Exception:                            # pragma: no cover - diagnostic only
            return
        key = attacker_state_key(live)
        if key is None:
            return
        self.keys_seen += 1
        fresh = _fresh_attacker_threat(live, None)
        seen = self._key_corpus.get(key)
        if seen is None:
            self._key_corpus[key] = fresh
            return
        self.key_repeats += 1
        if seen != fresh:
            differing = [f.name for f in dc_fields(seen)
                         if getattr(seen, f.name) != getattr(fresh, f.name)]
            self._fail(f"BELIEF KEY IS INCOMPLETE — two boards share attacker key {key!r} but "
                       f"differ on {differing}; the memo would serve the first board's belief for "
                       f"the second, silently changing Φ_belief {where}")

    def _check_turn(self, battle, state: _BattleState, curr_ctx: BattleContext,
                    legal=None) -> None:
        events = battle.events_since(state.prev_cursor)
        delta = TurnDelta.build_from_events(
            state.prev_ctx, curr_ctx, state.last_action, events)
        # Advance the shared clock first, exactly as EpisodeTracker.update_progress_clock does
        # (obs-side, before the reward reads `last_penalty`).
        if legal is not None:
            try:
                state.clock.update(delta, battle.strict_view().live, legal)
            except Exception as e:            # pragma: no cover - diagnostic only
                print(f"  [NOTICE] clock update skipped: {e}", flush=True)
        self.turns_compared += 1
        where = f"[{battle.battle_tag}] turn {battle.turn}"
        self._check_attacker_key(battle, where)

        for name, arm in state.arms.items():
            fast_r = arm.fast.process_turn_reward(battle, delta)
            full_r = arm.full.process_turn_reward(battle, delta)
            bf, bx = arm.fast._last_breakdown, arm.full._last_breakdown

            for field in RewardBreakdown.field_names():
                a = getattr(bf, field)
                b = getattr(bx, field)
                if not math.isfinite(a) or not math.isfinite(b):
                    self._fail(f"{name}: non-finite {field} fast={a!r} full={b!r} {where}")
                # (1) bit-identity — `!=`, deliberately not isclose
                if a != b:
                    self._fail(f"{name}: A FAST PATH CHANGED THE REWARD (the skip, the belief "
                               f"memo, or both) — {field} fast={a!r} full={b!r} {where}; "
                               f"active BIAS = {sorted(arm.fast._active_bias)}; memo = "
                               f"{arm.fast._belief_memo.stats()}")
                if b != 0.0:
                    self.activations[name][field] += 1
                # (2) the skip is exactly the suppression's complement
                if field in _BIAS_FIELDS and field not in arm.fast._active_bias and b != 0.0:
                    self._fail(f"{name}: term {field}={b!r} is CHARGED but the composition "
                               f"reports it inactive — the skip would drop a live term {where}")
            if fast_r != full_r:
                self._fail(f"{name}: total {fast_r!r} != {full_r!r} {where}")

    def _battle_finished_callback(self, battle) -> None:
        """Settle the terminal turn (win_loss fires there), then drop per-battle state."""
        tag = battle.battle_tag
        state = self._per_battle.get(tag)
        if state is None or state.prev_ctx is None or state.last_action is None:
            self._per_battle.pop(tag, None)
            return
        try:
            mask = np.ones(11, dtype=np.int8)
            curr_ctx = BattleContext.from_battle(
                battle, mask, state.our_slots, state.opp_slots)
            self._check_turn(battle, state, curr_ctx)
        except Exception as e:                     # pragma: no cover - diagnostic only
            print(f"  [NOTICE] {tag}: final-turn check skipped: {e}", flush=True)
        finally:
            # (5) roll this battle's memo counters up before the state is dropped — a memo that
            # never SERVED would make the identity claim above evidence about the cold path only.
            for arm in state.arms.values():
                st = arm.fast._belief_memo.stats()
                self.memo_hits += st["attacker_hits"] + st["row_hits"]
                self.memo_misses += st["attacker_misses"] + st["row_misses"]
            self._per_battle.pop(tag, None)

    def choose_move(self, battle):
        try:
            state = self._get_state(battle)
            legal = LegalActions.from_battle(battle)
            mask = Gen3ActionMasker.get_mask(battle, legal=legal)
            curr_ctx = BattleContext.from_battle(
                battle, mask, state.our_slots, state.opp_slots)

            if state.prev_ctx is not None and state.last_action is not None:
                self._check_turn(battle, state, curr_ctx, legal=legal)

            valid = np.where(mask == 1)[0]
            if len(valid) == 0:
                return self.choose_random_move(battle)
            choice = int(np.random.choice(valid))

            if not battle.finished:
                # Drive EVERY manager identically so all cross-turn state stays in lockstep —
                # a drifting counter would look exactly like a skip bug.
                for arm in state.arms.values():
                    arm.fast.record_action(curr_ctx, choice)
                    arm.full.record_action(curr_ctx, choice)
                state.prev_ctx = curr_ctx
                state.last_action = choice
                state.prev_cursor = battle.event_cursor

            return Gen3ActionMapper.action_to_order(choice, battle)
        except SystemExit:
            raise
        except Exception as e:
            print(f"\n[FUZZ FATAL] {battle.battle_tag} turn {battle.turn}: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)


def _report(player: RewardSkipParityPlayer, battles: int) -> int:
    print("\n" + "=" * 78)
    print(f"REWARD SKIP PARITY — {player.turns_compared} decisions over {battles} battles, "
          f"{len(COMPOSITIONS)} compositions")
    print("=" * 78)

    for name, cfg in COMPOSITIONS.items():
        comp = reward_class_composition(cfg)
        skipped = len(_BIAS_FIELDS) - comp["bias"]
        print(f"\n  {name:<24} {comp['terminal']} TERMINAL + {comp['pbrs']} PBRS + "
              f"{comp['bias']} BIAS  →  {skipped}/{len(_BIAS_FIELDS)} BIAS terms SKIPPED")

    cov = player.activations[_COVERAGE_ARM]
    print(f"\n  TRIGGER COVERAGE — BIAS terms that FIRED on the '{_COVERAGE_ARM}' arm.")
    print("  Same decisions as the other arms, so these are exactly the computations the")
    print("  production arm skipped. A zero here means the corpus never exercised that term.")
    print(f"  {'field':<24}{'turns non-zero':>16}{'':>6}")
    print("  " + "-" * 46)
    for field in _BIAS_FIELDS:
        flag = "" if cov[field] else "   (never seen)"
        req = " *" if field in _REQUIRED_COVERAGE else "  "
        print(f"  {field:<24}{cov[field]:>16}{req}{flag}")
    print("  " + "-" * 46)
    print("  * = required; a run that never fires these is not evidence and FAILS below.")

    # The mirror image: the ONE term the production composition still CHARGES must be seen to
    # fire, or this run never tested that the fast path keeps it.
    kept = player.activations["default"]["no_progress_tax"]
    print(f"\n  KEPT-TERM COVERAGE — 'default' arm charged no_progress_tax on {kept} turns")
    print("  (the single BIAS term production does NOT skip; 0 means the run never checked it).")

    total_lookups = player.memo_hits + player.memo_misses
    rate = (100.0 * player.memo_hits / total_lookups) if total_lookups else 0.0
    print(f"\n  BELIEF-MEMO COVERAGE — {player.memo_hits} SERVED / {total_lookups} lookups "
          f"({rate:.1f}% hit) across every arm.")
    print(f"  Attacker-key corpus: {player.keys_seen} boards, {player.key_repeats} repeat keys, "
          f"{len(player._key_corpus)} distinct.")
    print("  A repeat key is one differential test of key completeness (same key ⇒ same belief);")
    print("  a served hit is one decision whose Φ_belief came from the cache, not the formula.")

    missing = [f for f in _REQUIRED_COVERAGE if not cov[f]]
    if not kept:
        missing.append("no_progress_tax@default")
    if not player.memo_hits:
        missing.append("belief-memo hits (nothing was ever SERVED from the cache)")
    if not player.key_repeats:
        missing.append("attacker-key repeats (key completeness was never differentially tested)")
    if missing:
        print(f"\n❌ INCONCLUSIVE — {len(missing)} required signal(s) never fired: "
              f"{', '.join(missing)}. Raise the battle count; a clean run over a corpus that")
        print("   never exercised the skipped terms proves nothing.")
        return 1
    if player.violations:
        print(f"\n❌ {len(player.violations)} violation(s)")
        return 1
    print(f"\n✅ PASS — every field of every breakdown bit-identical with the fast paths (skip + "
          f"belief memo) ON vs OFF, across all {len(COMPOSITIONS)} compositions.")
    return 0


async def main(n_battles: int) -> int:
    ts = int(time.time()) % 100000
    pool = TeamLoader().get_sample_teams() or TeamLoader().get_all_teams()
    if not pool:
        print("no gen3ou teams found under data/teams", file=sys.stderr)
        return 1
    player = RewardSkipParityPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"RSPz{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"RSPo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False)

    print(f"Reward skip-parity fuzz — {BATTLE_FORMAT} — {n_battles} battles via the local "
          f"bridge (no server)", flush=True)
    await run_local_battles(player, opp, n_battles)
    return _report(player, n_battles)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sys.exit(asyncio.run(main(n)))
