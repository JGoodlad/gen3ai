"""Top-down per-decision **CPU** profiler for the trainer turn — no GPU, no live server.

The obs-build benchmark (`obs_build_benchmark.py`) is a microscope on ONE stage (the
observation encode). This is the wide-angle lens: it walks a real ``gen3ou`` battle
**in-process via the local BattleStream bridge** and times every CPU stage the training env
(`Gen3Env`) runs per decision, so you can see where a turn's controllable CPU actually goes.

Stages timed (all the CPU work `Gen3Env` owns per step — see `gen3_env.py`
embed_battle / calc_reward / action_to_order / step):

  * **parse + event-log fold** — the real `Gen3Battle.parse_message` (protocol → state +
    BattleEvent log), timed by subclassing the battle class the bridge feeds.
  * **obs build** — the shared `live_view()` board build + `get_mask` + `tracker.record` +
    `update_progress_clock` + `state_encoder.encode`. The `live_view()` line is timed on its
    own because it is memoized per state-epoch and five stages read it: without it, whichever
    stage runs first silently carries the whole build (see `_choose_move_timed`).
  * **reward** — `tracker.build_delta` (TurnDelta fold) + `reward_manager.process_turn_reward`.
  * **action map** — `Gen3ActionMapper.action_to_order`.
  * **tracker** — `advance` + `reward_manager.record_action`.

Three-bucket philosophy (matches what you asked for):
  * **Our CPU** — everything above. This is the headline number and the only thing we can
    optimize. Reported as a stacked per-decision breakdown.
  * **Must-pay overhead** — the Node sim advance + bridge IPC. It's an async subprocess wait,
    NOT our CPU, so it is deliberately NOT counted as a stage (you said you don't care about it).
  * **GPU / model forward** — EXCLUDED by design: a random legal action stands in for the
    policy, so no model is loaded and the GPU is never touched.

Reading the numbers: absolute µs/ms scale with machine load; the **per-stage shares** and the
**calls/decision** are the load-stable signal. Run on an idle box for a clean baseline.

Run directly (no server needed — local bridge):
    python src/agents/training/trainer_turn_benchmark.py                 # defaults
    python src/agents/training/trainer_turn_benchmark.py --decisions 200 --warmup 3 --seed 0
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import asyncio
import random
import shlex
import statistics
import sys
import time
import traceback
from typing import Dict, List

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions
from agents.observation.state_encoder import get_observation_encoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker
from agents.training.reward_manager import (
    Gen3RewardManager, RewardConfig, format_reward_composition)
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# Accumulated wall time (seconds) inside the real Gen3Battle.parse_message, across the whole
# run. Parse fires once per protocol line (many per turn), so it's summed globally and divided
# by decisions at the end.
_PARSE_SECONDS = [0.0]


class _TimedGen3Battle(Gen3Battle):
    """Gen3Battle that accumulates the wall time spent parsing the protocol stream + folding
    the event log — the 'parse' CPU stage. Behaviour is otherwise identical."""

    def parse_message(self, split_message):
        t0 = time.perf_counter()
        try:
            return super().parse_message(split_message)
        finally:
            _PARSE_SECONDS[0] += time.perf_counter() - t0


class _StageAcc:
    """Per-stage (total_seconds, call_count) accumulator."""

    def __init__(self):
        self.total: Dict[str, float] = {}
        self.count: Dict[str, int] = {}

    def add(self, stage: str, dt: float) -> None:
        self.total[stage] = self.total.get(stage, 0.0) + dt
        self.count[stage] = self.count.get(stage, 0) + 1


def _team_pool() -> list:
    loader = TeamLoader()
    pool = loader.get_sample_teams() or loader.get_all_teams()
    if not pool:
        raise RuntimeError("no gen3ou teams found under data/teams")
    return pool


class _TrainerTurnPlayer(Player):
    """Drives the full per-decision CPU path of Gen3Env (minus the GPU forward) on every
    decision and accumulates per-stage timings. A random legal action stands in for the
    policy, so no model is loaded."""

    def __init__(self, *args, warmup: int, use_assembler: bool = True,
                 reward_config: "RewardConfig | None" = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_assembler = use_assembler
        # The reward COMPOSITION under test. `None` = the dataclass default (the shaped
        # production reward). `--reward-argv` builds it through the REAL launch path
        # (`build_parser` → `RewardConfig.from_args`), so a composition measured here is a
        # composition a run can actually be launched with — never a hand-built lookalike.
        self._reward_config = reward_config
        maps = load_mappings()
        self.obs_enc = get_observation_encoder(maps)
        self._trackers: Dict[str, EpisodeTracker] = {}
        self._rewards: Dict[str, Gen3RewardManager] = {}
        self.acc = _StageAcc()
        self.decisions = 0           # all decisions (incl. warmup), used for parse/dec
        self.measured = 0            # non-warmup decisions counted into stage stats
        self._warmup = warmup
        # Robust per-decision distributions (ms), live decisions only:
        self._cpu_samples: List[float] = []     # our CPU inside choose_move (excl. parse)
        self._cycle_samples: List[float] = []    # full wall-clock between our decisions
        self._last_enter: Dict[str, float] = {}  # last live choose_move entry, per battle tag

    def _state_for(self, tag: str):
        tr = self._trackers.get(tag)
        if tr is None:
            tr = EpisodeTracker(history_cap=1)
            self._trackers[tag] = tr
            self._rewards[tag] = Gen3RewardManager(config=self._reward_config)  # QUIET
        return tr, self._rewards[tag]

    def choose_move(self, battle):
        try:
            return self._choose_move_timed(battle)
        except Exception:
            traceback.print_exc()
            return self.choose_random_move(battle)

    def _choose_move_timed(self, battle):
        tag = battle.battle_tag
        enter = time.perf_counter()           # wall-clock at the top of our decision
        tr, rm = self._state_for(tag)
        # Time stages only after warmup (lru_cache / encoder warm once globally); always
        # execute them so the tracker timeline + reward state stay consistent.
        live = self.decisions >= self._warmup
        a = self.acc
        dec_cpu = [0.0]                        # our CPU summed across this decision's stages

        def timed(stage, fn):
            if not live:
                return fn()
            t0 = time.perf_counter()
            r = fn()
            dt = time.perf_counter() - t0
            a.add(stage, dt)
            dec_cpu[0] += dt
            return r

        # --- obs build (Gen3Env.embed_battle) ---
        # ⚠️ ATTRIBUTION, and it is the whole reason this line exists (gen3_live_view_stage_v1):
        # FIVE stages below consume `battle.live_view()`, which memoizes per state-epoch — so
        # whichever stage happens to run FIRST pays the entire 12-mon build and the other four
        # read a warm slot. Until 2026-08-23 that stage was `obs: legal + mask`, which therefore
        # reported 22% of worker CPU while its OWN work (parse the request, set 11 bits, run two
        # integrity checks) measures 0.028 ms — 88% of the line was the shared board build
        # wearing the mask's name, and a whole optimization pass was aimed at it on that basis.
        # Building it here, explicitly and on its own line, is what makes the other five honest.
        timed("obs: live_view (shared build)", battle.live_view)
        # Build the LegalActions snapshot ONCE and share it between the mask and the mapper,
        # exactly as the env does (post-021f2d3).
        def _legal_and_mask():
            legal = LegalActions.from_battle(battle)
            return legal, Gen3ActionMasker.get_mask(battle, legal=legal)
        legal, mask = timed("obs: legal + mask", _legal_and_mask)
        if int(np.asarray(mask).sum()) == 0:
            return self.choose_random_move(battle)  # forced wait — not a measured decision
        timed("obs: tracker.record", lambda: tr.record(battle, mask=mask, legal=legal))
        # ⚠️ MEASUREMENT HONESTY, the same correction `obs_build_benchmark` took on 2026-08-16
        # and this one did not: until gen3_obs_assembler_v1 this benchmark called `encode` with
        # only the HP tracker, so the progress clock, the recency triplets, the H-A pair loop
        # and the H-B event-window write were all timed as SKIPPED — production pays every one
        # of them. The obs share below was therefore an UNDERSTATEMENT, and no earlier absolute
        # figure from this script is comparable to one printed after this date.
        timed("obs: update_progress_clock",
              lambda: tr.update_progress_clock(battle, legal))
        _kw = dict(hp_tracker=tr.hidden_power_tracker, legal=legal,
                   progress_clock=tr.progress_clock, recency=tr.recency,
                   pair_history=tr.pair_history, event_window=tr.event_window)
        if self._use_assembler:
            _kw["assembler"] = tr.obs_assembler(self.obs_enc.dimension)
        timed("obs: state_encoder.encode", lambda: self.obs_enc.encode(battle, **_kw))

        # --- reward (Gen3Env.calc_reward) ---
        delta = timed("reward: build_delta", lambda: tr.build_delta(battle=battle))  # noqa: E501
        timed("reward: process_turn_reward", lambda: rm.process_turn_reward(battle, delta))

        # --- action map (Gen3Env.action_to_order) + bookkeeping (Gen3Env.step) ---
        valid = [i for i, v in enumerate(np.asarray(mask)) if v]
        # `self._choice_rng` (not the `random` module) so this player's action picks depend only
        # on how many times IT has drawn — the bridge interleaves it with the opponent's draws.
        idx = self._choice_rng.choice(valid) if valid else 0
        order = timed("action map", lambda: Gen3ActionMapper.action_to_order(
            idx, battle, legal=legal, mask=mask))

        def _book():
            tr.advance(idx)
            rm.record_action(tr.last_ctx, idx)
        timed("tracker advance/record", _book)

        self.decisions += 1
        if live:
            self.measured += 1
            self._cpu_samples.append(dec_cpu[0] * 1e3)
            # Full per-decision wall-clock = entry-to-entry between consecutive LIVE decisions
            # (skips warmup + forced-wait calls, which don't update _last_enter). Everything
            # not in our CPU (sim advance, opponent move, parse, framework) lives in here.
            prev = self._last_enter.get(tag)
            if prev is not None:
                self._cycle_samples.append((enter - prev) * 1e3)
            self._last_enter[tag] = enter
        return order if order is not None else self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        tag = battle.battle_tag
        self._trackers.pop(tag, None)
        self._rewards.pop(tag, None)


# Group leaf stages into the buckets a reader thinks in.
_GROUPS = [
    ("parse + event-log fold", ["parse"]),
    # ⚠️ A stage that is TIMED but not listed here is silently dropped from the group total —
    # the group is summed from this list, not from the accumulator. `obs: update_progress_clock`
    # was added with gen3_obs_assembler_v1 and must be listed, or the obs share understates by
    # the whole tracker fold (progress clock + recency + H-A pair + H-B window).
    ("obs build", ["obs: live_view (shared build)", "obs: legal + mask", "obs: tracker.record",
                   "obs: update_progress_clock", "obs: state_encoder.encode"]),
    ("reward (TurnDelta fold)", ["reward: build_delta", "reward: process_turn_reward"]),
    ("action map", ["action map"]),
    ("tracker advance/record", ["tracker advance/record"]),
]


def _p(xs: List[float], q: float) -> float:
    """Percentile (q in [0,1]) of a sample list, robust to small N. 0.5 == median."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _report(player: _TrainerTurnPlayer, battles: int) -> None:
    a = player.acc
    measured = max(1, player.measured)
    # parse is summed over the whole stream; report per measured decision.
    parse_ms = (_PARSE_SECONDS[0] / max(1, player.decisions)) * 1e3
    leaf_ms = {s: (a.total[s] / measured) * 1e3 for s in a.total}
    leaf_ms["parse"] = parse_ms
    our_cpu = sum(leaf_ms.values())

    cpu_med = statistics.median(player._cpu_samples) if player._cpu_samples else 0.0
    our_cpu_med = cpu_med + parse_ms                      # add the (uniform) parse cost
    cyc_med = statistics.median(player._cycle_samples) if player._cycle_samples else 0.0
    wait_med = max(0.0, cyc_med - our_cpu_med)            # sim + opp + framework (bridge)
    cpu_pct = (our_cpu_med / cyc_med * 100) if cyc_med else 0.0

    print("\n" + "=" * 78)
    print(f"TRAINER-TURN PROFILE  ({player.measured} measured decisions, {battles} battles)")
    print("Random legal action replaces the policy forward → NO model, NO GPU, NO server.")
    print("=" * 78)

    # ---- Top-down: where does the wall-clock of one of OUR decisions go? ----
    print("\n  PER-DECISION WALL-CLOCK  (median; bridge = ONE in-process battle):")
    print(f"    full decision cycle          : {cyc_med:7.3f} ms   "
          f"(p90 {_p(player._cycle_samples, 0.9):.3f})")
    print(f"    └ our controllable CPU       : {our_cpu_med:7.3f} ms   "
          f"({cpu_pct:.0f}% of the cycle; p90 {_p(player._cpu_samples, 0.9)+parse_ms:.3f})")
    print(f"    └ sim + opp + framework      : {wait_med:7.3f} ms   "
          f"({100 - cpu_pct:.0f}%)  [MUST-PAY]")
    print("    CAVEAT: the bridge runs ONE battle in-process — its sim/framework wall is NOT")
    print("            the 64-env networked training pipeline (no batching/parallelism/sockets,")
    print("            no GPU sync). Trust OUR-CPU as representative; the wait fraction is")
    print("            bridge-only and indicative. The real FPS verdict is training-side.")

    # ---- Drill-down: where does OUR CPU go (the part we can optimize)? ----
    print("\n  OUR-CPU STAGE BREAKDOWN  (mean per decision):")
    print(f"  {'stage':<32}{'mean':>10}{'% our-CPU':>12}{'calls/dec':>12}")
    print("  " + "-" * 64)
    for label, leaves in _GROUPS:
        grp_ms = sum(leaf_ms.get(l, 0.0) for l in leaves)
        grp_pct = grp_ms / our_cpu * 100 if our_cpu else 0
        print(f"  {label:<32}{grp_ms:>8.3f}ms{grp_pct:>11.0f}%")
        if len(leaves) > 1:
            for l in leaves:
                if l in leaf_ms:
                    cpd = a.count.get(l, 0) / measured
                    print(f"    └ {l:<28}{leaf_ms[l]:>8.3f}ms{'':>12}{cpd:>11.2f}")
    print("  " + "-" * 64)
    print(f"  {'OUR CONTROLLABLE CPU / decision':<32}{our_cpu:>8.3f}ms  (mean)")
    print(f"  {'  · median / p90':<32}{cpu_med + parse_ms:>8.3f} / "
          f"{_p(player._cpu_samples, 0.9) + parse_ms:.3f} ms")
    print()
    print("  [model forward (GPU): excluded — random legal action stands in]")
    print("  NOTE: absolute ms scale with machine load; trust the % shares + calls/dec.")


def _battle_seed(seed: int, index: int) -> List[int]:
    """The gen-5 PRNG seed for battle ``index`` of a ``--pin-battles`` run — a pure function of
    ``(--seed, index)``, so arm A and arm B play the SAME ORDERED SEQUENCE of distinct battles.

    ⚠️ WITHOUT this the script cannot A/B anything inside a decision, and that is a measured
    fact rather than a caution: three back-to-back 400-decision runs on 2026-09-06 put the
    SHAPED arm's `process_turn_reward` at 0.145 ms and 0.102 ms while the TERMINAL-only arm
    between them read 0.111 ms — the run-to-run spread was LARGER than the effect and carried
    the wrong sign. Each invocation walks a fresh RANDOM battle, so two runs profile two
    different boards; the `--seed` alone pins the team draw and this player's action picks but
    NOT the sim dice. Same failure `live_view_build_benchmark` exists to avoid.

    Left OFF by default: an unpinned run samples the board distribution, which is what the
    headline share table wants. Turn it ON for any before/after or arm-vs-arm claim.
    """
    # 4 x 16-bit words, the sim's gen-5 seed shape. Distinct per index, deterministic in `seed`.
    h = (seed * 2654435761 + index * 40503 + 1) & 0xFFFFFFFFFFFFFFFF
    return [(h >> (16 * k)) & 0xFFFF for k in range(4)]


def _reward_config_from_argv(reward_argv: "str | None") -> "RewardConfig | None":
    """Build the reward config under test from a string of REAL `train_rl_agent` flags.

    e.g. ``--reward-argv '--no-hand-shaping --terminal-indicator --victory-value 1.0'``.
    It goes through `build_parser()` + `RewardConfig.from_args` — the launch path's own two
    steps — rather than constructing a `RewardConfig(...)` here by hand, so the arm this
    benchmark times is provably the arm a launch produces from the same flags. A hand-built
    lookalike is how a benchmark ends up measuring a composition nothing runs.

    ``None`` (the flag unset) keeps the dataclass default = the shaped production reward.
    Note this deliberately stops at `parse_args`: `resolve_config`'s cross-flag REFUSALS are a
    launch concern, and this profiler is allowed to time a composition the launcher would
    argue with (it never trains anything). The composition it resolved is PRINTED, so what
    was measured is on the record either way.
    """
    if reward_argv is None:
        return None
    from main.train_rl_agent import build_parser
    args = build_parser().parse_args(shlex.split(reward_argv))
    return RewardConfig.from_args(args)


async def main(target_decisions: int, battle_cap: int, warmup: int, seed: int,
               use_assembler: bool = True, reward_argv: "str | None" = None,
               pin: bool = False) -> int:
    # `random.seed(seed)` alone was NOT enough: four drawers shared the global stream (two
    # teambuilders, this player's action pick, and `RandomPlayer`'s entire policy) and the bridge
    # interleaves the two players' `choose_move` calls, so the draw ORDER — and therefore the
    # battle whose per-stage CPU is reported — was not reproducible from the seed. Each drawer now
    # gets its own stream off a distinct derived seed; p1 and p2 must differ or the two sides draw
    # in lockstep. The module seed stays for any other global consumer in the import graph.
    random.seed(seed)
    reward_config = _reward_config_from_argv(reward_argv)
    ts = int(time.time()) % 100000
    pool = _team_pool()
    player = _TrainerTurnPlayer(
        warmup=warmup, use_assembler=use_assembler, reward_config=reward_config,
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool, rng_seed=seed * 4 + 0),
        rng_seed=seed * 4 + 1,
        account_configuration=AccountConfiguration(f"TTz{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=_TimedGen3Battle)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool, rng_seed=seed * 4 + 2),
        rng_seed=seed * 4 + 3,
        account_configuration=AccountConfiguration(f"TTo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False)

    print(f"Trainer-turn CPU profiler — {BATTLE_FORMAT} — target {target_decisions} measured "
          f"decisions (warmup {warmup}, seed {seed}, no GPU/server, "
          f"obs-assembler {'ON' if use_assembler else 'OFF'}, "
          f"battles {'PINNED' if pin else 'random'})", flush=True)
    # The reward arm is part of the measurement, so it is printed with the run header rather
    # than left implicit — `format_reward_composition` is the SAME announcer a launch prints.
    print(format_reward_composition(reward_config or RewardConfig())
          + ("   [default — shaped production reward]" if reward_config is None else
             f"   [--reward-argv {reward_argv!r}]"), flush=True)
    battles = 0
    while player.measured < target_decisions and battles < battle_cap:
        # The bridge assigns each battle a process-unique tag (local_battle_runner._BATTLE_SEQ),
        # so repeated single-battle calls never collide on a reused tag — no per-call cleanup
        # needed; _battle_finished_callback drops each battle's tracker/reward as it ends.
        await run_local_battles(player, opp, 1, seed=_battle_seed(seed, battles) if pin else None)
        battles += 1

    if player.measured == 0:
        print("no decisions measured — bridge built? teams loaded?", file=sys.stderr)
        return 1
    _report(player, battles)
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Top-down per-decision trainer-turn CPU profiler.")
    p.add_argument("--decisions", type=int, default=400, dest="target_decisions",
                   help="measured (post-warmup) decisions to accumulate (default 400)")
    p.add_argument("--battle-cap", type=int, default=60,
                   help="max battles to play while accumulating (default 60)")
    p.add_argument("--warmup", type=int, default=3,
                   help="decisions to skip before timing (cache/encoder warmup) (default 3)")
    p.add_argument("--seed", type=int, default=0, help="action-selection seed (default 0)")
    p.add_argument("--pin-battles", dest="pin", action="store_true",
                   help="pin the SIM DICE so the battle SEQUENCE is a pure function of --seed. "
                        "REQUIRED for any before/after or arm-vs-arm comparison: unpinned, two "
                        "invocations play different battles and the run-to-run spread exceeds "
                        "the effect (measured). Off by default so the headline share table "
                        "still samples the board distribution.")
    p.add_argument("--reward-argv", default=None,
                   help="real train_rl_agent flags naming the reward composition to time, as "
                        "ONE quoted string (e.g. '--no-hand-shaping --terminal-indicator "
                        "--victory-value 1.0'). Parsed by the launcher's own build_parser + "
                        "RewardConfig.from_args. Unset = the shaped production default.")
    p.add_argument("--no-assembler", dest="use_assembler", action="store_false",
                   help="encode WITHOUT the incremental obs cache (gen3_obs_assembler_v1) — the "
                        "A/B arm. Run both back to back in one session: absolute ms on this box "
                        "are load-dependent, so only a same-load pair is a measurement.")
    return p.parse_args(argv)


if __name__ == "__main__":
    # A benchmark on a busy box reports a confidently wrong number — say so up front.
    from utils.contention import warn_if_contended
    warn_if_contended("trainer-turn benchmark")
    args = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(
        main(args.target_decisions, args.battle_cap, args.warmup, args.seed,
             args.use_assembler, args.reward_argv, args.pin)))
