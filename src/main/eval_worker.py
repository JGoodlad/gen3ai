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
frozen trainee (greedy) vs a pool snapshot (stochastic + the flat pool teambuilder unless
eval_sentinel_greedy, which makes the sentinel argmax AND gives it the trainee's own builder), loaded via
``load_opponent_snapshot`` (inference-only, ride-along keys ignored) and version-checked; a
fixed/ext_ opponent plays a foreign frozen model
(``load_foreign_opponent``, greedy yardstick). Under ``seed_rule = "per_game"`` (the Lane H gate
only) a policy opponent's decisions are logged and a SAMPLED sentinel draws the Rust eval core's keyed
draw (``install_opponent_log``), so the gate compares that regime game for game. Sentinel/fixed model
loads are CACHED per worker by path so a fine shard split doesn't pay an N× (~27MB) deserialize — the
snapshot is immutable within a cycle, so a cache hit is safe (the version check runs on the first, real
load).
"""
import os

# CPU eval shares the box with training — keep BLAS/OMP from spawning a thread per
# core in this process (mirrors the trainer's SubprocVecEnv workers).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import random
import asyncio
import hashlib
import functools
import traceback
from datetime import datetime

from sb3_contrib import MaskablePPO
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

from agents.inference.player import RLPlayer
from agents.model.snapshot import (current_model_version, load_opponent_snapshot,
                                   load_foreign_opponent, maybe_compile_extractor)
from agents.observation.state_encoder import load_mappings
from agents.training.eval_callback import (
    BATTLE_FORMAT, build_eval_opponents, build_eval_players, episode_length_sum,
    ForensicQuota,
)
from agents.training.eval_sharding import ShardedEvalPool, ShardResult, BOT, SENTINEL, FIXED
from agents.training.reward_manager import Gen3RewardManager, RewardConfig
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder, _install_team_rng


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


def _sentinel_tb(trainee_tb, opp_tb, sentinel_greedy: bool):
    """The pool SENTINEL's eval teambuilder — the trainee's own when the run is in the SYMMETRIC
    (greedy) regime, else the historical unbiased pool builder.

    🚨 THE ASYMMETRY THIS CLOSES WAS INVISIBLE AND SYSTEMATIC (gen3_eval_sentinel_greedy_default_v1,
    2026-09-07). The trainee draws from ``Gen3Teambuilder(all_teams, bias_teams=sample_teams,
    bias_prob=0.1)`` — a 10% tilt toward the curated sample teams — while the sentinel drew from the
    flat ``Gen3Teambuilder(all_teams)``. The dense snapshot ladder gives BOTH sides the biased
    builder, so an eval sentinel edge and a ladder edge for the SAME frozen pair were two different
    experiments; measured over the 60 pairs both sources covered on ``ai_v12_02_winprob_critic``,
    the eval edge favoured the newer snapshot by **+8.9 pp [+7.0, +10.7]**. The team asymmetry and
    the greedy-vs-stochastic asymmetry are two halves of ONE regime, which is why one switch moves
    both: four regimes would be three more than anyone wants to interpret, and the pair of them is
    exactly the condition under which the ladder may REUSE an eval-measured pair.

    ⚠️ SYMMETRIC HERE MEANS "the sentinel draws the way the TRAINEE does", which on a SPECIALIST run
    (``--trainee-team`` / ``--trainee-teams``) means the sentinel is pinned to the taught team(s)
    too — right for eval (both players in the distribution the run trains on), but NOT the ladder's
    own draw. The eval row records the ladder-comparability separately (see
    ``selfplay_callback``'s ``sentinel_regime.symmetric_teams``), so a specialist run's pairs are
    always replayed by the ladder rather than reused.

    ⚠️ THE TWO PLAYERS SHARE ONE BUILDER OBJECT, which is safe only because this worker constructs
    it with both of ``Gen3Teambuilder``'s stateful features at their defaults: ``team_pfsp`` is
    "off" (so ``_draw_team`` is one ``_rng.choice`` and nothing here ever calls
    ``record_team_pfsp_outcome``) and ``_block_episodes`` is 1 (``set_block_episodes`` is a
    TRAINING-env call, never made here), so ``yield_team`` carries no per-player state. If either
    ever reaches eval, give the sentinel its own instance — a shared block cache would hand both
    players the SAME team and burn the block twice as fast."""
    return trainee_tb if sentinel_greedy else opp_tb


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

    Amortizes the ~27MB ``load_opponent_snapshot`` / ``load_foreign_opponent`` deserialize across all
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


def unit_seed(seed_base: int, item_key: str, shard_index: int) -> int:
    """This shard UNIT's seed — a pure function of ``(seed_base, opponent, shard index)``.

    🚨 DELIBERATELY NOT a function of the worker id or the claim order. Work-stealing decides
    WHICH worker plays a shard, and that is a race: keying the dice off the worker would make a
    seeded cycle depend on the worker count, which is the one thing a seed exists to remove.
    Keyed this way, a shard's whole byte stream — its team draws, its scripted-bot rolls and its
    sim dice — is fixed by the PLAN, so ``--workers 1`` and ``--workers 8`` produce the identical
    cycle. What is NOT fixed is battle INTERLEAVE: at ``concurrency > 1`` several battles of one
    unit share the process-global `random` stream, and the order they draw from it is a timing
    race. Reproducibility therefore needs ``concurrency == 1``; the worker count is free.

    Only ever set by an OFFLINE generator (``main.ops.eval_trace_gen``). A live training eval
    passes no seed and is byte-identical to the pre-seed behaviour.
    """
    digest = hashlib.blake2b(f"{seed_base}:{item_key}:{shard_index}".encode(),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 62) - 1)


def seed_unit_streams(seed_base, item_key, shard_index, trainee_tb, opp_tb):
    """Pin every RNG a shard unit draws from, and return its bridge ``seed_base`` (or None).

    Three streams, because eval draws from three: the process-global `random` (the SCRIPTED bots'
    move choice), the two teambuilders' draw RNGs (which team each side pilots), and the sim's own
    PRNG (the dice), which is pinned per battle inside the bridge runner. The trainee and the
    greedy sentinel are argmax, so no torch stream needs pinning for them to repeat.
    """
    if seed_base is None:
        return None
    us = unit_seed(int(seed_base), item_key, shard_index)
    random.seed(us)
    # Distinct derived seeds so the two sides do not draw the same team sequence. (In the
    # SYMMETRIC greedy regime `_sentinel_tb` hands the sentinel the trainee's builder OBJECT, so
    # both sides then share one stream — unchanged, and still deterministic.)
    _install_team_rng(trainee_tb, us ^ 0x7472616E65650001)
    _install_team_rng(opp_tb, us ^ 0x6F70706F6E656E74)
    return us


def install_opponent_log(player, keyed: bool) -> None:
    """M5 Lane H, ``seed_rule = "per_game"`` only (the gate's worker; a live eval never sets it): log the
    POLICY OPPONENT's decisions, one ``[dec, action, argmax, margin]`` row per decision into
    ``player._keyed_log`` (``_play_per_game`` resets it and sets ``_keyed_seed`` before every game), so
    the gate compares a sentinel's / fixed opponent's own choices with the Rust eval core's — a greedy
    opponent's near-tie flip is then JUDGED rather than surfacing as a trainee divergence. ``dec`` is the
    index of the opponent's decision in the game (the core's ``dec_n``): one per ``choose_move`` that
    reaches the forward, whatever its stale re-decides. The margin is the top-2 legal log-prob margin
    (greedy) or the keyed draw's CDF margin.

    ``keyed`` (the SAMPLED sentinel regime, ``eval_sentinel_greedy`` off): the opponent draws the Rust
    eval core's KEYED draw instead of torch's global generator — the inverse-CDF action of its own legal
    log-probs at ``--self-play-temp``, at the uniform keyed by (the game's ``sample_seed``, the opponent
    stream, env 0, episode 0, ``dec``): exactly the executor's key, so the sampled regime is compared
    GAME FOR GAME. Through RLPlayer's opt-in ``_action_sampler`` hook (Lane G)."""
    import numpy as np
    import torch

    from agents.training import keyed_draw as KD

    player._keyed_seed = None
    player._keyed_log = []
    player._opp_dec = None
    player._opp_row = None
    player._keyed_margin = None

    def legal_logp(masked_logits):
        ml = masked_logits[0].detach().float().cpu()
        legal = (ml > -1e8).numpy()
        lp = torch.log_softmax(ml, -1).numpy()
        return np.where(legal, lp, -np.inf).astype(np.float32)

    def sample(masked_logits, temperature):
        if player._keyed_seed is None or player._opp_dec is None:
            raise RuntimeError("keyed sampler: no game key / decision index (per_game sets them)")
        lpm = legal_logp(masked_logits)[None]
        u = KD.keyed_uniforms(int(player._keyed_seed), KD.STREAM_OPPONENT, 0, 0, int(player._opp_dec))
        a, m = KD.keyed_actions(lpm, np.atleast_1d(u), float(temperature))
        player._keyed_margin = float(m[0])
        return int(a[0])

    orig_predict = player._predict_best_action
    orig_choose = player.choose_move

    def predict(battle, stochastic=False, need_aux=True, temperature=1.0):
        player._keyed_margin = None
        idx, probs, mask = orig_predict(battle, stochastic=stochastic, need_aux=need_aux, temperature=temperature)
        if idx is not None and player._opp_dec is not None:
            lp = legal_logp(player._last_masked_logits)
            srt = np.sort(lp[np.isfinite(lp)])
            margin = player._keyed_margin if player._keyed_margin is not None else (
                float(srt[-1] - srt[-2]) if srt.size > 1 else float("inf"))
            player._opp_row = [int(player._opp_dec), int(idx), int(np.argmax(lp)), float(margin)]
        return idx, probs, mask

    def choose_move(battle):
        player._opp_dec, player._opp_row = len(player._keyed_log), None
        try:
            return orig_choose(battle)
        finally:
            if player._opp_row is not None:
                player._keyed_log.append(player._opp_row)
            player._opp_dec = None

    player._predict_best_action = predict
    player.choose_move = choose_move
    if keyed:
        player._action_sampler = sample


#: The stream attribute of each bot RNG (``bot_inventory``'s streams → the player's attribute).
_BOT_STREAM_ATTR = {"choice": "_choice_rng", "protect": "_protect_rng", "bait": "_rng"}


def _play_per_game(unit, pool, trainee, opponent, item, seed_base, bridge_impl, game_log_path=None):
    """M5 Lane H — the PER-GAME seed rule (``rust_eval.seeds``, ``seed_rule = "per_game"``): each game of
    the unit is played alone, in plan order, on the in-process bridge, with its OWN battle seed and — for a
    scripted bot — its streams re-seeded exactly as the Rust eval core re-seeds them at the game's start.
    The two players' teams were pre-drawn by the same rule (``_per_game_teams``). Optional: one JSON line
    per game (winner, end turn, the trainee's actions and top-2 margins) into ``game_log_path``."""
    import random as _random

    from agents.training import mirrored_pairs as MP
    from agents.training.eval_sharding import BOT
    from agents.training.reward_weights import _TIMEOUT_TURN_CAP
    from agents.training.rust_eval import seeds as SD
    from agents.training.trace_result import classify_result

    mirrored = bool(getattr(pool, "mirrored", False))
    points: list = []
    trainee.decision_log = {} if game_log_path else None
    for g in pool.game_range(unit):
        # THE MIRRORED-PAIR RULE (`seeds.pair_game`): a pair's two games share the key of its first —
        # the same battle seed, bot streams, choice and sample seeds; `_per_game_teams` swapped the teams.
        key, _swapped = SD.pair_game(seed_base, item.key, g, mirrored)
        words = SD.battle_seed(key)
        if item.kind == BOT:
            for stream, s in SD.bot_stream_seeds(item.key, words).items():
                attr = _BOT_STREAM_ATTR[stream]
                if stream == "choice" or hasattr(opponent, attr):
                    setattr(opponent, attr, _random.Random(s))
        trainee._choice_rng = _random.Random(SD.derived_seed(key, "p1choice"))
        logged = getattr(opponent, "_keyed_log", None) is not None     # a policy opponent (install_opponent_log)
        if logged:
            opponent._keyed_seed = SD.sample_seed(key)
            opponent._keyed_log = []
        before = set(trainee._battles)
        asyncio.run(run_local_battles(trainee, opponent, 1, concurrency=1, impl=bridge_impl, seed=list(words)))
        tag = next((t for t in trainee._battles if t not in before), None)
        b = trainee._battles[tag] if tag is not None else None
        if b is not None and b.finished:
            res, _kind = classify_result(won=b.won, lost=b.lost, finished=True, turn=b.turn,
                                         turn_cap=_TIMEOUT_TURN_CAP)
            points.append(MP.result_points(res))
        else:
            points.append(None)                      # an unfinished game voids its PAIR, never half of it
        if game_log_path:
            assert b is not None
            dl = trainee.decision_log.get(tag, [])
            with open(game_log_path, "a") as f:
                f.write(json.dumps({"item": item.key, "game": g, "shard": unit.shard_index,
                                    "winner": 1 if b.won else (2 if b.lost else 0), "end_turn": int(b.turn),
                                    "actions": [d[0] for d in dl], "margins": [d[1] for d in dl],
                                    "logp": [d[2] for d in dl],
                                    "opp": list(opponent._keyed_log) if logged else None,
                                    "seed": list(words),
                                    **({"swapped": _swapped} if mirrored else {})}) + "\n")
    return MP.pair_counts(points) if mirrored else None


def _per_game_teams(unit, pool, item, seed_base, trainee_tb, opp_builder):
    """The unit's teams under the per-game seed rule, as two ``SequenceTeambuilder``s (plan order).

    MIRRORED (``pool.mirrored``): a pair's two games draw the SAME two teams (the key of its first game)
    and the second game hands them over — the trainee pilots what the opponent drew and vice versa."""
    from agents.training.rust_eval import seeds as SD
    from main.rust_core_cutover.envs import SequenceTeambuilder

    mirrored = bool(getattr(pool, "mirrored", False))
    ours, theirs = [], []
    for g in pool.game_range(unit):
        k, swapped = SD.pair_game(seed_base, item.key, g, mirrored)
        a, b = SD.draw_team(trainee_tb, k, SD.TRAINEE), SD.draw_team(opp_builder, k, SD.OPPONENT)
        ours.append(b if swapped else a)
        theirs.append(a if swapped else b)
    return SequenceTeambuilder(ours), SequenceTeambuilder(theirs)


async def _play(trainee, opponent, n_games, use_bridge, concurrency, bridge_impl="node",
                seed_base=None):
    if use_bridge:
        await run_local_battles(trainee, opponent, n_games, concurrency=concurrency,
                                impl=bridge_impl, seed_base=seed_base)
    else:
        if seed_base is not None:
            raise ValueError("seeded eval needs the in-process bridge — a Showdown SERVER mints "
                             "its own dice and no client can pin them.")
        await trainee.battle_against(opponent, n_battles=n_games)


def _play_unit(unit, pool, model, opp_model_cache, current_version, trainee_tb, opp_tb,
               mappings, server_config, concurrency, device, model_dir, step, tag, wid,
               use_bridge, gamma, self_play_temp, sentinel_greedy, reward_factory,
               bridge_impl="node", compile_extractor=False,
               forensic_quota: "ForensicQuota | None" = None,
               seed_base=None, seed_rule: str = "unit", game_log_path=None) -> ShardResult:
    """Play one shard unit and return its RAW (additive) result.

    A fresh trainee + opponent are built per unit so the measurement (win count, reward sum, δ
    pool, forensic capture) is independent and the parent can pool it exactly. The opponent MODEL
    (sentinel/fixed) is cached; only the cheap player wrapper is rebuilt per unit. ``reward_factory``
    is the run's reward (from ``model_config.json``) so eval MEASURES with the trained reward."""
    item = unit.item
    n_games = unit.n_games

    # Pin this unit's dice FIRST — before any player is built, because construction itself draws
    # (a scripted bot's setup, a teambuilder's first pick). Returns None for a live eval, which
    # sets nothing and leaves every stream exactly where it was.
    per_game = seed_rule == "per_game"
    if getattr(pool, "mirrored", False) and not per_game:
        raise ValueError("a MIRRORED plan needs seed_rule per_game (a pair shares one battle seed, so every "
                         "game must be seeded) — the eval callback sets it; refusing to play the pairs unpaired")
    if per_game:
        if seed_base is None or not use_bridge or int(concurrency) != 1:
            raise ValueError("seed_rule per_game needs a seed_base, the in-process bridge and concurrency 1")
        opp_builder = (opp_tb if item.kind == BOT else
                       _sentinel_tb(trainee_tb, opp_tb, sentinel_greedy) if item.kind == SENTINEL else
                       _fixed_opponent_tb(item, opp_tb))
        trainee_tb, _unit_opp_tb = _per_game_teams(unit, pool, item, seed_base, trainee_tb, opp_builder)
        opp_tb = _unit_opp_tb
        unit_seed_base = None
    else:
        unit_seed_base = seed_unit_streams(seed_base, item.key, unit.shard_index, trainee_tb, opp_tb)

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
            lambda: load_opponent_snapshot(item.path, current_version=current_version,
                                           device=device),
            compile_extractor=compile_extractor, device=device)
        opponent = RLPlayer(
            model=opp_model, team=opp_tb if per_game else _sentinel_tb(trainee_tb, opp_tb, sentinel_greedy),
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config, mappings=mappings,
            account_configuration=AccountConfiguration(f"SPse{tag}", "password"),
            max_concurrent_battles=concurrency,
            stochastic=not sentinel_greedy, temperature=self_play_temp,
            start_listening=not use_bridge)
        if per_game:
            install_opponent_log(opponent, keyed=not sentinel_greedy)
    elif item.kind == FIXED:
        opp_model = _get_opponent_model(
            opp_model_cache, item.path,
            lambda: load_foreign_opponent(item.path, current_version=current_version,
                                          device=device, config_path=item.config_path)[0],
            compile_extractor=compile_extractor, device=device)
        fixed_tb = opp_tb if per_game else _fixed_opponent_tb(item, opp_tb)
        opponent = RLPlayer(
            model=opp_model, team=fixed_tb, battle_format=BATTLE_FORMAT,
            server_configuration=server_config, mappings=mappings,
            account_configuration=AccountConfiguration(f"SOop{tag}", "password"),
            max_concurrent_battles=concurrency,
            stochastic=False, temperature=1.0,  # eval = greedy yardstick
            start_listening=not use_bridge)
        if per_game:
            install_opponent_log(opponent, keyed=False)
    else:  # pragma: no cover - guarded by EvalItem.__post_init__
        raise ValueError(f"unknown item kind {item.kind!r}")

    # Forensic capture writes into the per-opponent dir; `trace_tag` namespaces this shard's files
    # so concurrent shards of the same opponent never collide. Per-unit quota is scaled down by the
    # shard count so the total traces per opponent stay ~bounded near the global cap.
    forensic_dir = (os.path.join(model_dir, "eval_traces", f"step_{step}", item.key)
                    if model_dir else None)
    n_shards = pool.shard_count(item.key)
    per_unit = ForensicQuota.coerce(forensic_quota).per_shard(n_shards)
    trainee.begin_forensic_cycle(
        forensic_dir, step, trace_tag=f"s{unit.shard_index}_",
        win_quota=per_unit.win, loss_quota=per_unit.loss, draw_quota=per_unit.draw)

    start = datetime.now()
    pair_counts = None
    if per_game:
        pair_counts = _play_per_game(unit, pool, trainee, opponent, item, seed_base, bridge_impl,
                                     game_log_path=game_log_path)
    else:
        asyncio.run(_play(trainee, opponent, n_games, use_bridge, concurrency, bridge_impl,
                          seed_base=unit_seed_base))
    dur = (datetime.now() - start).total_seconds()

    res = ShardResult(
        unit_id=unit.unit_id, item_key=item.key, worker_id=wid,
        n_won=trainee.n_won_battles, n_finished=trainee.n_finished_battles,
        sum_reward=trainee.episode_reward_sum, n_episodes=trainee.n_reward_episodes,
        sum_ep_len=episode_length_sum(trainee), duration_sec=dur,
        td_residuals=trainee.td_residuals(),
        # What the forensic QUOTA actually kept from this shard, so the per-cycle manifest can
        # state the trace SELECTION rather than leaving every consumer to assume it was uniform.
        traces_written=trainee.traces_written, traces_won=trainee.traces_won,
        # DRAWS: every drawn battle PLAYED (`n_drawn`) and how many the draw quota KEPT. The
        # played count is here and nowhere else — poke-env books a tie as neither a win nor a
        # loss and a 250-turn timeout as our forfeit, so without this the draw rate cannot be
        # recovered from the shard record at all.
        n_drawn=trainee.draws_seen, traces_drawn=trainee.traces_drawn,
        # MIRRORED pairs: the unit's pentanomial (None for an unmirrored plan).
        pair_counts=pair_counts)
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
    # No default that means a REGIME: the callback always writes this key (it is resolved and
    # recorded per run). `False` is the pre-2026-09-07 shape and is kept only so a cfg written
    # by an older tree still parses.
    sentinel_greedy = cfg.get("eval_sentinel_greedy", False)
    # OFFLINE generation only (`main.ops.eval_trace_gen`). ABSENT — the live case — means every
    # stream is left alone and this worker is byte-identical to the pre-seed one.
    seed_base = cfg.get("seed_base")
    # M5 Lane H: "per_game" = the Rust eval core's per-GAME seed rule (the gate's Python path);
    # absent = today's per-UNIT rule (an offline generation) or no seed at all (a live eval).
    seed_rule = cfg.get("seed_rule", "unit")
    game_log_path = cfg.get("game_log_path")
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
    # TRAINED with (terminal_indicator / draw_penalty / …). Threaded to every EvalRLPlayer below; a bare
    # default here once silently scored eval with the wrong reward.
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
    # The run's capture quota, as the parent configured it. ABSENT means an older parent, which
    # is the DEFAULT quota — never zero, which would silently capture nothing.
    forensic_quota = ForensicQuota.coerce(cfg.get("forensic_quota"))
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
            compile_extractor, forensic_quota, seed_base=seed_base, seed_rule=seed_rule,
            game_log_path=game_log_path)
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
