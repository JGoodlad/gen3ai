"""Subprocess eval worker.

Loads a frozen model snapshot, runs the per-opponent eval gather for the opponents
it work-steals from the shared pool against the (shared) Showdown server, and
writes one result JSON per opponent that the trainer's ``PerOpponentEvalCallback``
picks up. Spawned by that callback as::

    python -m main.eval_worker <config.json>

Running eval in a fresh process means all its memory is returned to the OS on
exit (no fragmentation in the trainer), and the frozen snapshot lets eval run in
parallel with training — the worker reads a static copy, not the mutating model.

Config JSON keys: snapshot, port, model_dir, step, n_games, opponent_pool,
claim_dir, result_dir, concurrency, device, worker_id, cycle_tag.

Optional self-play keys (absent → pure bot eval, byte-identical to before):
  sentinels       — [{label, path, step}] pool snapshots to play as opponents
  self_play_temp  — sampling temperature for the (stochastic) sentinel opponents
  fixed_opponents — [{label, path, config_path}] stable cross-run opponents (ext_<label>), played
                    GREEDY via _eval_fixed (loaded by load_foreign_opponent, arch-gated only)

Work stealing: all workers share the claim universe (bot names + sentinel labels)
+ `claim_dir`; each repeatedly claims the next unclaimed item (atomic O_EXCL lock)
and writes its result to `result_dir/result__<item>.json`, until the pool is
exhausted. A worker that finishes a cheap item immediately grabs the next, so uneven
per-item cost self-balances across the (default 3) workers. Bot items run the bot
roster path; sentinel items play the frozen trainee (greedy) vs the pool snapshot
(stochastic, version-checked on load).
"""
import os

# CPU eval shares the box with training — keep BLAS/OMP from spawning a thread per
# core in this process (mirrors the trainer's SubprocVecEnv workers).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import asyncio
import traceback

from sb3_contrib import MaskablePPO
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_model_snapshot, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from agents.training.eval_callback import (
    BATTLE_FORMAT, EvalRLPlayer, build_eval_opponents, build_eval_players,
    eval_one_matchup, run_eval, claim_next_opponent,
)
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def _run(cfg: dict) -> None:
    # Rebuild the stateless eval infrastructure deterministically from the data dir,
    # exactly as train_rl_agent does — nothing is passed in-memory across the process
    # boundary except the config (and the snapshot zip on disk).
    mappings = load_mappings()
    loader = TeamLoader()
    all_teams = loader.get_all_teams()
    sample_teams = loader.get_sample_teams()
    trainee_tb = Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)
    opp_tb = Gen3Teambuilder(all_teams)

    port = cfg.get("port")
    server_config = localhost_server_configuration(port) if port else LocalhostServerConfiguration
    # Bridge eval: play in-process via run_local_battles (no server). Players are built
    # start_listening=False; server_config is then inert (the bridge supplies the transport).
    use_bridge = cfg.get("use_showdown_bridge", False)
    concurrency = cfg["concurrency"]
    device = cfg.get("device", "cpu")
    n_games = cfg["n_games"]
    model_dir = cfg.get("model_dir")
    step = cfg["step"]

    # Frozen trainee weights — inference only, so the base algorithm + env=None is enough
    # (the policy class and its extractor kwargs are restored from the zip).
    model = MaskablePPO.load(cfg["snapshot"], env=None, device=device)

    bot_names = cfg["opponent_pool"]
    sentinels = cfg.get("sentinels") or []
    sentinel_by_label = {s["label"]: s for s in sentinels}
    # Stable cross-run opponents (ext_<label>): foreign frozen models played as a fixed yardstick.
    fixed_opponents = cfg.get("fixed_opponents") or []
    fixed_by_label = {f["label"]: f for f in fixed_opponents}
    self_play_temp = cfg.get("self_play_temp", 1.0)
    # Run's discount for the recorder's TD-residual δ (#4); the parent passes model.gamma.
    gamma = cfg.get("gamma", 0.99)
    # Eval the pool sentinels greedy (best-vs-best) instead of stochastic when set.
    sentinel_greedy = cfg.get("eval_sentinel_greedy", False)
    # One combined work-steal universe: bot names + sentinel labels + stable-opponent labels.
    claim_universe = list(bot_names) + [s["label"] for s in sentinels] \
        + [f["label"] for f in fixed_opponents]
    # Build the CURRENT-code version once, for any load that needs an arch check (sentinels via
    # check_compatible, stable opponents via check_opponent_compatible).
    current_version = current_model_version(mappings) if (sentinels or fixed_opponents) else None

    claim_dir = cfg["claim_dir"]
    result_dir = cfg["result_dir"]
    wid = cfg["worker_id"]
    cycle_tag = cfg["cycle_tag"]

    claim_seq = 0
    while True:
        name = claim_next_opponent(claim_dir, claim_universe)
        if name is None:
            break  # every item claimed by some worker → this one is done
        # Unique account suffix per (cycle, worker, claim) so the lingering
        # connection from a prior claim can't collide on the shared server.
        tag = f"{cycle_tag}{wid}{claim_seq}"
        claim_seq += 1

        if name in sentinel_by_label:
            result = _eval_sentinel(
                model, sentinel_by_label[name], current_version, trainee_tb, opp_tb,
                mappings, server_config, concurrency, device, self_play_temp,
                n_games, model_dir, step, tag, use_bridge, gamma, sentinel_greedy,
            )
        elif name in fixed_by_label:
            result = _eval_fixed(
                model, fixed_by_label[name], current_version, trainee_tb, opp_tb,
                mappings, server_config, concurrency, device,
                n_games, model_dir, step, tag, use_bridge, gamma,
            )
        else:
            opponents = build_eval_opponents(
                server_config, opp_tb, [name], tag, start_listening=not use_bridge)
            players = build_eval_players(
                model, [name], trainee_tb, mappings, server_config, concurrency, tag,
                start_listening=not use_bridge, gamma=gamma,
            )
            m = asyncio.run(run_eval(players, opponents, n_games, model_dir, step,
                                     use_bridge=use_bridge,
                                     bridge_concurrency=(concurrency if use_bridge else 1)))
            result = {
                "win_rate": m["win_rates"][name],
                "reward_mean": m["reward_means"][name],
                "ep_len": m["ep_lens"][name],
                "duration_sec": m["durations_sec"][name],
            }
            # TD-residual tail (#4) — present only if this opponent had captured battles.
            if name in m.get("td_resid_tails", {}):
                result["td_resid_tail"] = m["td_resid_tails"][name]
        result["worker_id"] = wid
        # Write atomically (tmp + rename) so the parent never reads a half-written file.
        out = os.path.join(result_dir, f"result__{name}.json")
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, out)


def _eval_trainee_vs_model(model, opp_model, label, trainee_tb, opp_tb, mappings, server_config,
                           concurrency, n_games, model_dir, step, tag, *, trainee_prefix,
                           opp_prefix, opp_stochastic, opp_temperature,
                           use_bridge=False, gamma=0.99) -> dict:
    """Play the frozen trainee (GREEDY) vs one already-loaded opponent model — one matchup.

    The shared core of the pool-sentinel and stable-opponent eval paths: those two differ ONLY in
    how the opponent model is loaded (``load_model_snapshot`` vs ``load_foreign_opponent``) and the
    opponent's action regime. The trainee is an ``EvalRLPlayer`` so it tracks reward + writes
    forensic traces, exactly like a bot matchup; it is always greedy (a stable win-rate signal).
    Returns the standard result dict — the caller adds any extras (e.g. the sentinel step)."""
    trainee = EvalRLPlayer(
        model=model, team=trainee_tb, battle_format=BATTLE_FORMAT,
        server_configuration=server_config, mappings=mappings,
        account_configuration=AccountConfiguration(f"{trainee_prefix}{tag}", "password"),
        max_concurrent_battles=concurrency,
        stochastic=False,  # eval measures the GREEDY policy → stable win-rate signal
        start_listening=not use_bridge, gamma=gamma,
    )
    opponent = RLPlayer(
        model=opp_model, team=opp_tb, battle_format=BATTLE_FORMAT,
        server_configuration=server_config, mappings=mappings,
        account_configuration=AccountConfiguration(f"{opp_prefix}{tag}", "password"),
        max_concurrent_battles=concurrency,
        stochastic=opp_stochastic, temperature=opp_temperature,
        start_listening=not use_bridge,
    )
    m = asyncio.run(eval_one_matchup(trainee, opponent, n_games, model_dir, step, label,
                                     use_bridge=use_bridge,
                                     bridge_concurrency=(concurrency if use_bridge else 1)))
    out = {"win_rate": m["win_rate"], "reward_mean": m["reward_mean"],
           "ep_len": m["ep_len"], "duration_sec": m["duration_sec"]}
    if m.get("td_resid_tail") is not None:  # #4 — present only if captured battles ran
        out["td_resid_tail"] = m["td_resid_tail"]
    return out


def _eval_sentinel(model, spec, current_version, trainee_tb, opp_tb, mappings,
                   server_config, concurrency, device, temperature,
                   n_games, model_dir, step, tag, use_bridge=False, gamma=0.99,
                   sentinel_greedy=False) -> dict:
    """Play the frozen trainee (greedy) vs one pool sentinel — one matchup.

    The sentinel is loaded via ``load_model_snapshot`` against the pool's shared
    ``model_config.json`` (a stale-arch snapshot fails with ``ModelVersionError``). By default it
    is **stochastic** (at ``temperature``), mirroring how it behaves as a TRAINING opponent; with
    ``sentinel_greedy`` it plays **argmax**, so the matchup is greedy-vs-greedy and
    ``win_rate_vs_pool`` / the snapshot ELO carry no temperature handicap.
    """
    sentinel_model = load_model_snapshot(
        spec["path"], env=None, current_version=current_version, device=device)
    out = _eval_trainee_vs_model(
        model, sentinel_model, spec["label"], trainee_tb, opp_tb, mappings, server_config,
        concurrency, n_games, model_dir, step, tag,
        trainee_prefix="SPtr", opp_prefix="SPse",
        opp_stochastic=not sentinel_greedy, opp_temperature=temperature,
        use_bridge=use_bridge, gamma=gamma)
    out["sentinel_step"] = spec.get("step")
    return out


def _eval_fixed(model, spec, current_version, trainee_tb, opp_tb, mappings,
                server_config, concurrency, device, n_games, model_dir, step, tag,
                use_bridge=False, gamma=0.99) -> dict:
    """Play the frozen trainee (greedy) vs one STABLE cross-run opponent — one matchup.

    Like ``_eval_sentinel`` but the opponent is a foreign frozen model from another run, loaded via
    ``load_foreign_opponent`` (gated on the OBSERVATION FAMILY only — same ``arch_signature`` — not
    the full ``check_compatible`` the pool sentinels use). In EVAL the opponent plays **greedy
    (temp 0)** — a fixed, deterministic yardstick gives the cleanest win-rate signal (during
    TRAINING the same opponent plays stochastic, to be less exploitable). ``ext_`` results are kept
    out of ``win_rate_vs_bots`` / the ELO fit by the parent callback.
    """
    opp_model, _foreign = load_foreign_opponent(
        spec["path"], current_version=current_version, device=device,
        config_path=spec.get("config_path"))
    return _eval_trainee_vs_model(
        model, opp_model, spec["label"], trainee_tb, opp_tb, mappings, server_config,
        concurrency, n_games, model_dir, step, tag,
        trainee_prefix="SOtr", opp_prefix="SOop",
        opp_stochastic=False, opp_temperature=1.0,  # eval = greedy (temp 0) → clean yardstick
        use_bridge=use_bridge, gamma=gamma)


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
