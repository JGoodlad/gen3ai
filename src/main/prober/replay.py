"""Counterfactual replay-to-end orchestration (Feature 2) — build the players, RELOAD the real
opponent, run the Monte-Carlo, aggregate into a win-probability.

The faithful single-line/multi-line driver is :func:`utils.bridge.counterfactual.replay_counterfactual`
(the scripted-prefix bridge runner); this module wires it to the prober: it maps the substitute action
index → a sim choice string (the same ``map_actions_at`` mapper the falsifier/lookahead use), builds a
GREEDY trainee ``RLPlayer`` from the loaded checkpoint, RESOLVES the recorded opponent (a reproducible
bot, an explicit checkpoint override, or — when neither is available — the trainee's own model as a
flagged self-play approximation), and loops the rollouts (each a fresh dice reseed) into a win-rate ±
Wilson CI. The session owns the (cached) model loads and passes them in.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from agents.training.obs_materializer import materialize_from_record
from main.prober.falsifier import _label_of, fresh_seeds
from utils.bridge.counterfactual import replay_counterfactual as _run_one
from utils.bridge.reconstruction import ReconstructionRecord

_OTHER = {"p1": "p2", "p2": "p1"}


def wilson_ci(wins: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    """Wilson score interval for a binomial win-rate — the right small-N CI (a normal approximation
    gives [0,0] at 0/k). ``(0.0, 1.0)`` for n == 0."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _bot_classes() -> dict:
    from agents.training.eval_callback import _EVAL_OPPONENT_SPECS
    return {name: cls for (name, cls, _prefix) in _EVAL_OPPONENT_SPECS}


def _model_obs_dim(model) -> "int | None":
    """The obs-vector dim the loaded policy was TRAINED on (its observation_space ``observation`` key),
    or None when it can't be introspected. Used to catch a cross-obs-version checkpoint before the play
    loop hits a confusing torch shape error."""
    try:
        return int(model.observation_space["observation"].shape[0])
    except (KeyError, AttributeError, TypeError, IndexError):
        return None


def build_trainee(play_model, record: ReconstructionRecord, mappings, server_config, *, tag: str):
    """A GREEDY ``RLPlayer`` (the measured policy) on the trainee's recorded side, transport-less for
    the in-process bridge. The team is forced to the recorded one by the runner; passed here too so
    construction is valid."""
    from poke_env.ps_client import AccountConfiguration
    from agents.inference.player import RLPlayer

    side = record.side_of(record.trainee_username)
    return RLPlayer(
        model=play_model, team=record.packed_team(side), battle_format=record.format_id,
        server_configuration=server_config, mappings=mappings,
        account_configuration=AccountConfiguration(f"CfT{tag}", "pw"),
        max_concurrent_battles=1, stochastic=False, start_listening=False)


#: A reloaded checkpoint opponent stands in for a POOL SENTINEL, and the eval worker plays
#: sentinels STOCHASTIC at temp 1.0 unless the run set ``--eval-sentinel-greedy`` (which
#: defaults False, and is False in every run on file). So "recorded" means stochastic here.
_RECORDED_CKPT_STOCHASTIC = True


def build_opponent(opp_name: str, record: ReconstructionRecord, play_model, mappings, server_config,
                   *, opponent_ckpt=None, opp_model=None, opponent_source: str = "auto",
                   opponent_stochastic: "Optional[bool]" = None, tag: str):
    """Resolve + build the opponent player for the continuation, RELOADING the real opponent where
    possible. Returns ``(player, source_label)``. Priority: an explicit checkpoint (``opp_model``) →
    a reproducible bot (the recorded opponent name is in the eval roster) → the trainee's own model
    (a self-play APPROXIMATION, flagged ``self_model_approx`` — the honest fallback when the real
    opponent isn't reloadable, e.g. an aged-out self-play sentinel).

    ``opponent_stochastic`` sets the SAMPLING REGIME of a model-backed opponent. ``None`` (the
    default) means **the regime the record says was played** — stochastic for a checkpoint
    opponent, because that is what ``eval_worker`` recorded for a pool sentinel
    (``stochastic = not eval_sentinel_greedy``, and ``eval_sentinel_greedy`` defaults False).
    Pass ``True``/``False`` to override deliberately.

    ⚠️ This used to be hard-wired GREEDY, and that was not a stylistic choice — a greedy copy of a
    net is strictly stronger than a temp-1.0 sample of it, so every Monte-Carlo label taken against
    a reloaded sentinel was biased LOW and every predicted-minus-MC gap biased HIGH, *in the
    direction of the hypothesis being tested*. Measured on 477 sentinel states (G0, 2026-08-22):
    re-labelling in the recorded regime moved MC by **+0.037 [+0.007, +0.066]**."""
    from poke_env.ps_client import AccountConfiguration
    from agents.inference.player import RLPlayer

    side = _OTHER[record.side_of(record.trainee_username)]
    team = record.packed_team(side)

    def _rl(model, stochastic, label):
        return RLPlayer(
            model=model, team=team, battle_format=record.format_id,
            server_configuration=server_config, mappings=mappings,
            account_configuration=AccountConfiguration(f"CfO{tag}", "pw"),
            max_concurrent_battles=1, stochastic=stochastic, start_listening=False), label

    if opponent_source not in ("auto", "bot", "self", "ckpt"):
        raise ValueError(f"unknown opponent_source {opponent_source!r}")
    if opponent_source == "ckpt" and opp_model is None:
        # Mirror the explicit `bot` raise — an explicit ckpt request must not silently fall through
        # to the self-model approximation.
        raise ValueError("--opponent-source ckpt requires --opponent-ckpt")

    if opp_model is not None and opponent_source in ("auto", "ckpt"):
        stochastic = (_RECORDED_CKPT_STOCHASTIC if opponent_stochastic is None
                      else bool(opponent_stochastic))
        regime = "stochastic" if stochastic else "greedy"
        return _rl(opp_model, stochastic, f"ckpt_{regime}:{opponent_ckpt}")

    bots = _bot_classes()
    if opponent_source in ("auto", "bot") and opp_name in bots:
        cls = bots[opp_name]
        player = cls(
            battle_format=record.format_id, team=team, server_configuration=server_config,
            account_configuration=AccountConfiguration(f"CfB{tag}", "pw"),
            max_concurrent_battles=1, start_listening=False)
        return player, f"bot:{opp_name}"

    if opponent_source == "bot":
        raise ValueError(f"opponent {opp_name!r} is not a known bot (roster: {sorted(bots)})")

    # Fallback: the trainee's own model plays the opponent side (self-play approximation). Stochastic,
    # mirroring how a self-play opponent acts. Flagged so the caller can surface the caveat.
    return _rl(play_model, True, "self_model_approx")


def replay_counterfactual_battle(
    record: ReconstructionRecord,
    summary: dict,
    npz: dict,
    inv_index: int,
    action: int,
    *,
    play_model,
    opp_name: str,
    mappings,
    server_config,
    opponent_ckpt=None,
    opp_model=None,
    opponent_source: str = "auto",
    opponent_stochastic: Optional[bool] = None,
    n_rollouts: int = 1,
    narrate: bool = False,
    impl: str = "node",
) -> dict:
    """Run the counterfactual: substitute ``action`` at ``inv_index``'s turn and play to a terminal
    ``n_rollouts`` times (each a fresh post-divergence dice reseed when ``n_rollouts`` > 1), returning
    a win-rate ± Wilson CI. See the module header for opponent resolution. Raises (like the falsifier)
    on a non-``move_selection`` anchor, an illegal substitute action, or a missing ``actions`` array.

    ``impl`` (``"node"`` default | ``"rust"``) selects the engine for BOTH legs this call spans: the
    offline replay driver that materializes the anchor, and the in-process sim bridge the live
    post-divergence rollouts run on."""
    invs = summary.get("invocations", [])
    if not (0 <= inv_index < len(invs)):
        raise IndexError(f"inv {inv_index} out of range (battle has {len(invs)})")
    inv = invs[inv_index]
    if inv.get("phase") != "move_selection":
        raise ValueError(
            f"inv {inv_index} is a {inv.get('phase')!r} decision — the divergence anchors at a "
            f"start-of-turn move round; pick that turn's move_selection invocation")
    if "actions" not in npz:
        raise ValueError("states.npz has no 'actions' array — only bridge-eval traces are replayable")
    if not record.trainee_username:
        raise ValueError("reconstruction record lacks trainee_username")
    turn = int(inv["turn"])

    actions = np.asarray(npz["actions"], dtype=int)
    chosen_idx = int(actions[inv_index])
    trace = materialize_from_record(record, actions=actions, mappings=mappings,
                                    map_actions_at=inv_index, stop_after_decision=inv_index,
                                    impl=impl)
    # Obs-version guard: the players build obs with the CURRENT encoder, but the resolved checkpoint
    # (nearest/recent tier) may have been trained on a different obs dim → a confusing torch shape error
    # deep in the play loop. Fail loud up front (mirrors the analyze path's obs_mismatch).
    _exp = _model_obs_dim(play_model)
    _got = len(trace.decisions[-1].obs) if trace.decisions else None
    if _exp and _got and _exp != _got:
        raise ValueError(
            f"obs-version mismatch: this trace's obs is {_got} dims but the loaded trainee model expects "
            f"{_exp} — replay a model trained on the current obs layout (use --ckpt / the exact tier)")
    choice_map = trace.action_choices or {}
    if int(action) not in choice_map:
        raise ValueError(f"action {action} is not legal at inv {inv_index} "
                         f"(legal actions: {sorted(choice_map)})")
    substitute_choice = choice_map[int(action)]

    seed = record.start_options().get("seed")
    post_seeds: List[Optional[str]] = (
        [None] if n_rollouts <= 1
        else list(fresh_seeds(n_rollouts, salt=f"{record.battle_tag}:{inv_index}:cf")))

    outcomes: List[str] = []
    source_label = None
    win_traj = loss_traj = None      # the first winning / first losing rollout's play-by-play (narrate)
    for k, ps in enumerate(post_seeds):
        trainee = build_trainee(play_model, record, mappings, server_config, tag=f"{inv_index}x{k}")
        opponent, source_label = build_opponent(
            opp_name, record, play_model, mappings, server_config, opponent_ckpt=opponent_ckpt,
            opp_model=opp_model, opponent_source=opponent_source,
            opponent_stochastic=opponent_stochastic, tag=f"{inv_index}x{k}")
        # Capture the move-by-move trajectory only until we have one win + one loss example (cheap).
        cap = narrate and (win_traj is None or loss_traj is None)
        res = _run_one(record, trainee=trainee, opponent=opponent, divergence_turn=turn,
                       substitute_choice=substitute_choice, seed=seed, post_t_seed=ps,
                       capture_trajectory=cap, impl=impl)
        outcomes.append(res["outcome"])
        if cap and res.get("trajectory"):
            if res["outcome"] == "win" and win_traj is None:
                win_traj = res["trajectory"]
            elif res["outcome"] == "loss" and loss_traj is None:
                loss_traj = res["trajectory"]

    n = len(outcomes)
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    lo, hi = wilson_ci(wins, n)

    caveats = [
        "The opponent plays its policy FRESH past the divergence; the recorded opponent's logged "
        "choices are invalid once our move changes (a different turn-T desyncs them).",
        "Most informative for THROWN-LATE losses; a matchup-lost-from-turn-1 game is rarely "
        "recoverable by one move (see the positional-grind decomposition).",
    ]
    if (source_label or "").startswith("ckpt_greedy"):
        caveats.insert(0, "Opponent = a reloaded checkpoint played GREEDY — STRONGER than the "
                          "temp-1.0 sample the eval worker recorded for a pool sentinel, so this "
                          "win-rate is biased LOW (measured +0.037 [+0.007, +0.066] over 477 "
                          "sentinel states). Drop --opponent-regime to play the recorded regime.")
    if source_label == "self_model_approx":
        caveats.insert(0, "Opponent = the trainee's OWN model (self-play APPROXIMATION) — the recorded "
                          "opponent was not a known bot and no --opponent-ckpt was given.")
    if n == 1:
        caveats.append("Single realized-dice line (n_rollouts=1) — NOT a probability. Raise "
                       "--rollouts for a Monte-Carlo win-rate ± CI over resampled post-divergence dice.")

    return {
        "inv": inv_index,
        "turn": turn,
        "trainee_side": record.side_of(record.trainee_username),
        "recorded_result": ((summary.get("meta") or {}).get("result") or "").lower() or None,
        "chosen": {"action": chosen_idx, "label": _label_of(inv, chosen_idx),
                   "choice": choice_map.get(chosen_idx)},
        "substitute": {"action": int(action), "label": _label_of(inv, int(action)),
                       "choice": substitute_choice},
        "opponent_source": source_label,
        "n_rollouts": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / n, 4) if n else None,
        "win_rate_ci": [round(lo, 4), round(hi, 4)],
        "outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
        "deterministic_line": n == 1,
        "winning_trajectory": win_traj,    # the play-by-play of a recovered WIN (narrate=True only)
        "losing_trajectory": loss_traj,
        "caveats": caveats,
    }
