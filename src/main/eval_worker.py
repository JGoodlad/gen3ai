"""Subprocess eval worker (battle-level work-stealing).

Loads a frozen model snapshot and **work-steals shard units** from the cycle's shared pool —
each unit is a chunk of one opponent's games, so an idle worker can drain a straggler's
remaining games instead of one worker grinding a whole opponent alone (the long-tail fix). For
each claimed unit it plays the games, reads the trainee's RAW counters (won/finished, reward sum,
turn sum, the raw per-decision δ samples) and publishes one ``shard__<unit_id>.json``; the parent
pools an opponent's shards back into one exact result. Spawned by the eval callbacks as::

    python -m main.eval_worker <config.json>

Running eval in a fresh process means all its memory is returned to the OS on exit (no
fragmentation in the trainer), and the frozen snapshot lets eval run in parallel with training —
the worker reads a static copy, not the mutating model.

The WHAT-to-play (the opponent items + shard plan) is read from the cycle's ``plan.json`` via
``ShardedEvalPool.from_plan`` — written once by the parent, the single source of truth, so the
worker never reconstructs the universe itself. The config JSON carries only the HOW (runtime):
snapshot, port, model_dir, step, claim_dir, result_dir, concurrency, device, worker_id, cycle_tag,
gamma, and the self-play knobs (self_play_temp, eval_sentinel_greedy).

Opponent kinds (from the plan item): a bot plays the scripted roster path; a sentinel plays the
frozen trainee (greedy) vs a pool snapshot (stochastic unless eval_sentinel_greedy), loaded via
``load_model_snapshot`` and version-checked; a fixed/ext_ opponent plays a foreign frozen model
(``load_foreign_opponent``, greedy yardstick). Sentinel/fixed model loads are CACHED per worker by
path so a fine shard split doesn't pay an N× (~27MB) deserialize — the snapshot is immutable within
a cycle, so a cache hit is safe (the version check runs on the first, real load).
"""
import os

# CPU eval shares the box with training — keep BLAS/OMP from spawning a thread per
# core in this process (mirrors the trainer's SubprocVecEnv workers).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import math
import asyncio
import functools
import traceback
from datetime import datetime

from sb3_contrib import MaskablePPO
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

from agents.inference.player import RLPlayer
from agents.model.snapshot import (current_model_version, load_model_snapshot,
                                   load_foreign_opponent, maybe_compile_extractor)
from agents.observation.state_encoder import load_mappings
from agents.training.eval_callback import (
    BATTLE_FORMAT, build_eval_opponents, build_eval_players, episode_length_sum,
    _FORENSIC_WIN_QUOTA, _FORENSIC_LOSS_QUOTA,
)
from agents.training.eval_sharding import ShardedEvalPool, ShardResult, BOT, SENTINEL, FIXED
from agents.training.reward_manager import Gen3RewardManager, RewardConfig
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def _build_trainee_tb(cfg: dict, all_teams, sample_teams):
    """The TRAINEE's eval teambuilder. When the run pins the trainee to one team
    (``--trainee-team`` → ``cfg['trainee_team_str']``, the raw Showdown export), eval MUST measure
    the model piloting THAT team — the worker used to hardcode the default full-pool builder here,
    so every specialist run's eval (win rates, ELO, vs-ext verdicts) measured the model piloting
    RANDOM teams it never trained on (pure out-of-distribution; the ai_v7_05–08 "plateau" was this
    gap, not the training). No pin → the default pool builder, byte-identical to the old behavior."""
    team_str = cfg.get("trainee_team_str")
    if team_str:
        # a LIST = the distillation/multi-team case (sample among the taught teams, as training does);
        # a plain str = the single --trainee-team pin.
        return Gen3Teambuilder(list(team_str) if isinstance(team_str, (list, tuple)) else [team_str])
    return Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)


def _fixed_opponent_tb(item, opp_tb):
    """A FIXED (stable/exploiter) opponent's eval teambuilder. A specialist opponent is MEASURED
    piloting ITS OWN pinned team(s) (``item.team_strs``, threaded from ``FixedOpponentEntry.to_cfg`` —
    the fold-back contract), so eval matches the training mix — the same eval-vs-training
    consistency rule as the trainee's own pin above. No pin → the shared pool builder."""
    team_strs = getattr(item, "team_strs", None)
    if team_strs:
        return Gen3Teambuilder(list(team_strs))   # multi-team specialist samples among ITS OWN teams
    team_str = getattr(item, "team_str", None)    # back-compat: older cfgs carry only the single pin
    if team_str:
        return Gen3Teambuilder([team_str])
    return opp_tb


def _get_opponent_model(cache: dict, path: str, loader, compile_extractor: bool = False,
                        device: str = "cpu"):
    """Return the opponent model for ``path``, loading it once per worker and caching it.

    Amortizes the ~27MB ``load_model_snapshot`` / ``load_foreign_opponent`` deserialize across all
    of an item's shards (and across shards of distinct items that share a path). Safe to cache: the
    snapshot at ``path`` is a frozen file, immutable for the cycle; the version check is part of the
    first real load, so a cache hit can't smuggle in an incompatible model.

    The compile rides this same cache, so it is paid at most once per distinct opponent — and since
    `torch.compile` keys on the CODE OBJECT, only the FIRST opponent in a worker pays; later ones
    hit the in-process dynamo cache in ~0s."""
    if path not in cache:
        model = loader()
        maybe_compile_extractor(model, compile_extractor,
                                label=f"eval-opp:{os.path.basename(path)}",
                                hide_cuda=str(device).startswith("cpu"))
        cache[path] = model
    return cache[path]


async def _play(trainee, opponent, n_games, use_bridge, concurrency, bridge_impl="node"):
    if use_bridge:
        await run_local_battles(trainee, opponent, n_games, concurrency=concurrency,
                                impl=bridge_impl)
    else:
        await trainee.battle_against(opponent, n_battles=n_games)


def _play_unit(unit, pool, model, opp_model_cache, current_version, trainee_tb, opp_tb,
               mappings, server_config, concurrency, device, model_dir, step, tag, wid,
               use_bridge, gamma, self_play_temp, sentinel_greedy, reward_factory,
               bridge_impl="node", compile_extractor=False) -> ShardResult:
    """Play one shard unit and return its RAW (additive) result.

    A fresh trainee + opponent are built per unit so the measurement (win count, reward sum, δ
    pool, forensic capture) is independent and the parent can pool it exactly. The opponent MODEL
    (sentinel/fixed) is cached; only the cheap player wrapper is rebuilt per unit. ``reward_factory``
    is the run's reward (from ``model_config.json``) so eval MEASURES with the trained reward."""
    item = unit.item
    n_games = unit.n_games

    # One EvalRLPlayer (greedy trainee, reward + forensic tracking), account unique per claim.
    trainee = build_eval_players(
        model, [item.key], trainee_tb, mappings, server_config, concurrency, tag,
        start_listening=not use_bridge, gamma=gamma, reward_fn_factory=reward_factory)[item.key]

    if item.kind == BOT:
        opponent = build_eval_opponents(
            server_config, opp_tb, [item.key], tag, start_listening=not use_bridge)[0][1]
    elif item.kind == SENTINEL:
        opp_model = _get_opponent_model(
            opp_model_cache, item.path,
            lambda: load_model_snapshot(item.path, env=None,
                                        current_version=current_version, device=device),
            compile_extractor=compile_extractor, device=device)
        opponent = RLPlayer(
            model=opp_model, team=opp_tb, battle_format=BATTLE_FORMAT,
            server_configuration=server_config, mappings=mappings,
            account_configuration=AccountConfiguration(f"SPse{tag}", "password"),
            max_concurrent_battles=concurrency,
            stochastic=not sentinel_greedy, temperature=self_play_temp,
            start_listening=not use_bridge)
    elif item.kind == FIXED:
        opp_model = _get_opponent_model(
            opp_model_cache, item.path,
            lambda: load_foreign_opponent(item.path, current_version=current_version,
                                          device=device, config_path=item.config_path)[0],
            compile_extractor=compile_extractor, device=device)
        fixed_tb = _fixed_opponent_tb(item, opp_tb)
        opponent = RLPlayer(
            model=opp_model, team=fixed_tb, battle_format=BATTLE_FORMAT,
            server_configuration=server_config, mappings=mappings,
            account_configuration=AccountConfiguration(f"SOop{tag}", "password"),
            max_concurrent_battles=concurrency,
            stochastic=False, temperature=1.0,  # eval = greedy yardstick
            start_listening=not use_bridge)
    else:  # pragma: no cover - guarded by EvalItem.__post_init__
        raise ValueError(f"unknown item kind {item.kind!r}")

    # Forensic capture writes into the per-opponent dir; `trace_tag` namespaces this shard's files
    # so concurrent shards of the same opponent never collide. Per-unit quota is scaled down by the
    # shard count so the total traces per opponent stay ~bounded near the global cap.
    forensic_dir = (os.path.join(model_dir, "eval_traces", f"step_{step}", item.key)
                    if model_dir else None)
    n_shards = pool.shard_count(item.key)
    trainee.begin_forensic_cycle(
        forensic_dir, step, trace_tag=f"s{unit.shard_index}_",
        win_quota=max(1, math.ceil(_FORENSIC_WIN_QUOTA / n_shards)),
        loss_quota=max(1, math.ceil(_FORENSIC_LOSS_QUOTA / n_shards)))

    start = datetime.now()
    asyncio.run(_play(trainee, opponent, n_games, use_bridge, concurrency, bridge_impl))
    dur = (datetime.now() - start).total_seconds()

    res = ShardResult(
        unit_id=unit.unit_id, item_key=item.key, worker_id=wid,
        n_won=trainee.n_won_battles, n_finished=trainee.n_finished_battles,
        sum_reward=trainee.episode_reward_sum, n_episodes=trainee.n_reward_episodes,
        sum_ep_len=episode_length_sum(trainee), duration_sec=dur,
        td_residuals=trainee.td_residuals(),
        # What the forensic QUOTA actually kept from this shard, so the per-cycle manifest can
        # state the trace SELECTION rather than leaving every consumer to assume it was uniform.
        traces_written=trainee.traces_written, traces_won=trainee.traces_won)
    win_rate = res.n_won / res.n_finished if res.n_finished else 0.0
    print(f"  {unit.unit_id}: {win_rate * 100:.1f}% ({res.n_won}/{res.n_finished})  "
          f"reward_sum={res.sum_reward:.1f}  [{dur:.0f}s]")
    return res


def _run(cfg: dict) -> None:
    # Rebuild the stateless eval infrastructure deterministically from the data dir, exactly as
    # train_rl_agent does — nothing is passed in-memory across the process boundary except the
    # config (and the snapshot zips + plan.json on disk).
    mappings = load_mappings()
    loader = TeamLoader()
    all_teams = loader.get_all_teams()
    sample_teams = loader.get_sample_teams()
    trainee_tb = _build_trainee_tb(cfg, all_teams, sample_teams)
    opp_tb = Gen3Teambuilder(all_teams)

    port = cfg.get("port")
    server_config = localhost_server_configuration(port) if port else LocalhostServerConfiguration
    use_bridge = cfg.get("use_showdown_bridge", False)
    # Which in-process bridge child: "node" (default) or "rust". Only meaningful when
    # use_bridge; threaded from the callback's base_cfg alongside use_showdown_bridge.
    bridge_impl = cfg.get("bridge_impl", "node")
    concurrency = cfg["concurrency"]
    device = cfg.get("device", "cpu")
    model_dir = cfg.get("model_dir")
    step = cfg["step"]
    gamma = cfg.get("gamma", 0.99)
    self_play_temp = cfg.get("self_play_temp", 1.0)
    sentinel_greedy = cfg.get("eval_sentinel_greedy", False)
    claim_dir = cfg["claim_dir"]
    result_dir = cfg["result_dir"]
    wid = cfg["worker_id"]
    cycle_tag = cfg["cycle_tag"]

    # Frozen trainee weights — inference only, so the base algorithm + env=None is enough.
    model = MaskablePPO.load(cfg["snapshot"], env=None, device=device)
    # The trainee plays EVERY eval game, so it is the hottest forward in this process. Same frozen
    # CPU B=1 shape as a training opponent => the same ~6.5x. Unlike an env worker this is a fresh
    # `Popen`d process (not forked from the trainer's forkserver), so it cannot inherit a compiled
    # graph — but it does hit the shared on-disk Inductor cache the trainer already warmed, and one
    # worker plays hundreds of games, so the compile pays back many times over.
    compile_extractor = bool(cfg.get("compile_extractor", False))
    maybe_compile_extractor(model, compile_extractor, label="eval-trainee",
                            hide_cuda=str(device).startswith("cpu"))

    # The trainee's reward factory — built from the RUN's model_config.json (the single source of
    # truth the version check already records), so eval MEASURES with the same reward the policy was
    # TRAINED with (bias_redesign / draw_penalty / …). Threaded to every EvalRLPlayer below; a bare
    # default here once silently scored eval with the wrong (bias_redesign=False) reward.
    _reward_cfg = {}
    if model_dir:
        try:
            with open(os.path.join(model_dir, "model_config.json")) as _f:
                _reward_cfg = json.load(_f)
        except (OSError, ValueError):
            _reward_cfg = {}
    reward_factory = functools.partial(Gen3RewardManager,
                                       config=RewardConfig.from_dict(_reward_cfg))

    # The shard plan (items + shard_games) is the parent's single source of truth — read it, don't
    # rebuild it. Build a current-code version only if some item needs an arch check on load.
    pool = ShardedEvalPool.from_plan(result_dir)
    needs_version = any(it.kind in (SENTINEL, FIXED) for it in pool.items)
    # Gate snapshot loads against THIS run's arch (belief-ON / popart / …), threaded from the parent
    # via the cfg — else a belief-ON self-play run FATALs on its own sentinels (check_compatible).
    current_version = (
        current_model_version(mappings, **cfg.get("arch_toggles", {})) if needs_version else None
    )
    opp_model_cache: dict[str, object] = {}

    claim_seq = 0
    while True:
        unit = pool.claim_next(claim_dir)
        if unit is None:
            break  # every unit claimed by some worker → this one is done
        # Unique account suffix per (cycle, worker, claim) so a lingering connection from a prior
        # claim can't collide on the shared server.
        tag = f"{cycle_tag}{wid}{claim_seq}"
        claim_seq += 1
        res = _play_unit(
            unit, pool, model, opp_model_cache, current_version, trainee_tb, opp_tb,
            mappings, server_config, concurrency, device, model_dir, step, tag, wid,
            use_bridge, gamma, self_play_temp, sentinel_greedy, reward_factory, bridge_impl,
            compile_extractor)
        pool.publish(result_dir, res)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m main.eval_worker <config.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    try:
        _run(cfg)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
