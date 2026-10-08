"""Counterfactual replay-to-end orchestration (Feature 2) — resolve the opponent, run the Monte-Carlo on
the RUST CORE, aggregate into a win-probability.

The play-out is :func:`utils.rust_env.counterfactual.replay_counterfactual` (``gen3_cf_core_playout_v1``,
poke-env retirement P6): ONE in-process ``play_out`` from the start of the divergence turn, the opponent's
RECORDED turn-T choice fed, OUR substitute the forced first action, then every rollout to the end. This
module wires it to the prober: it maps the substitute action index → its sim choice string (the core
walk's tokens, ``core_walk.decision_choices`` — the mapper falsify / lookahead use), RESOLVES the
recorded opponent (an explicit checkpoint, a reproducible in-core BOT, or — when neither is available —
the trainee's own model as a flagged self-play approximation), and folds the rollouts (each a fresh
post-divergence dice reseed) into a win-rate ± Wilson CI. The session owns the (cached) model loads.

REPRODUCIBLE BY CONSTRUCTION: every draw is seeded from the battle and the decision — the dice
(``fresh_seeds(…, salt=f"{tag}:{inv}:cf")``, as before), a stochastic model opponent's sampler (one
``torch.Generator`` per rollout, :func:`sampler_seeds`) and a bot's streams (:func:`bot_seed`) — so the
same question asked twice answers the same. (The poke-env road drew a sampled opponent's actions and a
bot's coins from the process-wide unseeded streams.)
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional

import numpy as np

from main.prober import core_walk
from main.prober.falsifier import _label_of, fresh_seeds
from utils.rust_env import counterfactual as CF
from utils.rust_env import successors as S

_OTHER = {"p1": "p2", "p2": "p1"}

#: The play-out's engine, stamped on every result (``--impl`` does not apply: there is one engine).
ENGINE = "rust_core"


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


def _bot_names() -> "list[str]":
    """The eval roster's bot names — every one is an in-core Rust port (``bots::Kind``)."""
    from agents.training.eval_schedule import eval_opponent_names

    return eval_opponent_names()


def _model_obs_dim(model) -> "int | None":
    """The obs-vector dim the loaded policy was TRAINED on (its observation_space ``observation`` key),
    or None when it can't be introspected. Used to catch a cross-obs-version checkpoint before the play
    loop hits a confusing torch shape error."""
    try:
        return int(model.observation_space["observation"].shape[0])
    except (KeyError, AttributeError, TypeError, IndexError):
        return None


def _salt_int(salt: str, nbytes: int) -> int:
    return int.from_bytes(hashlib.sha256(salt.encode()).digest()[:nbytes], "big")


def sampler_seeds(battle_tag: str, inv_index: int, n: int) -> "list[int]":
    """Rollout ``k``'s stochastic-opponent sampler seed (a ``torch.Generator`` seed), from the battle
    and the decision — shared by every substitute of that decision, like the dice."""
    return [_salt_int(f"{battle_tag}:{inv_index}:cf:sampler:{k}", 7) for k in range(n)]


def bot_seed(battle_tag: str, inv_index: int) -> int:
    """The in-core bot's route seed (< 2^53) — rollout ``k``'s streams are
    ``bot_stream_seed(bot_seed, k, s)`` (``rust_env_opponents``)."""
    return _salt_int(f"{battle_tag}:{inv_index}:cf:bot", 6)


#: A reloaded checkpoint opponent stands in for a POOL SENTINEL, and the eval worker plays
#: sentinels STOCHASTIC at temp 1.0 unless the run set ``--eval-sentinel-greedy``. So "recorded"
#: means stochastic here.
_RECORDED_CKPT_STOCHASTIC = True


def resolve_opponent(opp_name: str, *, play_model, opponent_ckpt=None, opp_model=None,
                     opponent_source: str = "auto", opponent_stochastic: "Optional[bool]" = None) -> "tuple[dict, str]":
    """Resolve who plays the opponent past the divergence, RELOADING the real opponent where possible.
    Returns ``(spec, source_label)``; ``spec`` is ``{"kind": "model", "model", "stochastic"}`` or
    ``{"kind": "bot", "name"}``. Priority: an explicit checkpoint (``opp_model``) → a reproducible bot
    (the recorded opponent name is in the eval roster; played by the IN-CORE Rust port) → the trainee's
    own model (a self-play APPROXIMATION, flagged ``self_model_approx`` — the honest fallback when the
    real opponent isn't reloadable, e.g. an aged-out self-play sentinel).

    ``opponent_stochastic`` sets the SAMPLING REGIME of a model-backed opponent. ``None`` (the
    default) means **the regime the record says was played** — stochastic for a checkpoint
    opponent, because that is what ``eval_worker`` recorded for a pool sentinel. Pass
    ``True``/``False`` to override deliberately.

    ⚠️ This used to be hard-wired GREEDY, and that was not a stylistic choice — a greedy copy of a
    net is strictly stronger than a temp-1.0 sample of it, so every Monte-Carlo label taken against
    a reloaded sentinel was biased LOW and every predicted-minus-MC gap biased HIGH, *in the
    direction of the hypothesis being tested*. Measured on 477 sentinel states (G0, 2026-08-22):
    re-labelling in the recorded regime moved MC by **+0.037 [+0.007, +0.066]**."""
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
        return {"kind": "model", "model": opp_model, "stochastic": stochastic}, f"ckpt_{regime}:{opponent_ckpt}"

    bots = _bot_names()
    if opponent_source in ("auto", "bot") and opp_name in bots:
        return {"kind": "bot", "name": opp_name}, f"bot:{opp_name}"

    if opponent_source == "bot":
        raise ValueError(f"opponent {opp_name!r} is not a known bot (roster: {sorted(bots)})")

    # Fallback: the trainee's own model plays the opponent side (self-play approximation). Stochastic,
    # mirroring how a self-play opponent acts. Flagged so the caller can surface the caveat.
    return {"kind": "model", "model": play_model, "stochastic": True}, "self_model_approx"


def replay_counterfactual_battle(
    record,
    summary: dict,
    npz: dict,
    inv_index: int,
    action: int,
    *,
    play_model,
    opp_name: str,
    opponent_ckpt=None,
    opp_model=None,
    opponent_source: str = "auto",
    opponent_stochastic: Optional[bool] = None,
    n_rollouts: int = 1,
    narrate: bool = False,
    impl: str = "rust",
    core=None,
) -> dict:
    """Run the counterfactual: substitute ``action`` at ``inv_index``'s turn and play to a terminal
    ``n_rollouts`` times (each a fresh post-divergence dice reseed when ``n_rollouts`` > 1), returning
    a win-rate ± Wilson CI. See the module header for opponent resolution. Raises (like the falsifier)
    on a non-``move_selection`` anchor, an illegal substitute action, or a missing ``actions`` array.

    The play-out runs on the in-process RUST CORE whatever ``impl`` says (there is no other engine for
    it since P6); ``impl="node"`` is not an error — it is said in the ``caveats``."""
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
    our_side = record.side_of(record.trainee_username)

    actions = np.asarray(npz["actions"], dtype=int)
    chosen_idx = int(actions[inv_index])
    choice_map, walk_turn = core_walk.decision_choices(record, our_side, inv_index)
    if int(walk_turn) != turn:
        raise ValueError(f"inv {inv_index}: the core walk reads this decision at turn {walk_turn}, the trace "
                         f"at turn {turn} — the record and the summary disagree")
    if int(action) not in choice_map:
        raise ValueError(f"action {action} is not legal at inv {inv_index} "
                         f"(legal actions: {sorted(choice_map)})")
    substitute_choice = choice_map[int(action)]

    own_core = core is None
    core = core or S.SearchCore()
    try:
        # Obs-version guard: the rows are the CURRENT core encoder's, but the resolved checkpoint
        # (nearest/recent tier) may have been trained on a different obs dim → a confusing torch shape
        # error deep in the play loop. Fail loud up front (mirrors the analyze path's obs_mismatch).
        _exp = _model_obs_dim(play_model)
        if _exp and _exp != core.obs_dim:
            raise ValueError(
                f"obs-version mismatch: the core's rows are {core.obs_dim} dims but the loaded trainee model "
                f"expects {_exp} — replay a model trained on the current obs layout (use --ckpt / the exact tier)")

        n = max(1, int(n_rollouts))
        tag = record.battle_tag or ""
        post_seeds: List[Optional[str]] = (
            [None] if n <= 1 else list(fresh_seeds(n, salt=f"{tag}:{inv_index}:cf")))
        spec, source_label = resolve_opponent(
            opp_name, play_model=play_model, opponent_ckpt=opponent_ckpt, opp_model=opp_model,
            opponent_source=opponent_source, opponent_stochastic=opponent_stochastic)
        our_policy = CF.ModelPolicy(play_model, stochastic=False)
        opp_policy = opp_bot = None
        stall_sides = [our_side]
        if spec["kind"] == "model":
            opp_policy = CF.ModelPolicy(spec["model"], stochastic=spec["stochastic"],
                                     seeds=sampler_seeds(tag, inv_index, n) if spec["stochastic"] else None)
            stall_sides.append(_OTHER[our_side])      # an RLPlayer opponent stall-forfeits too
        else:
            opp_bot = {"name": spec["name"], "seed": bot_seed(tag, inv_index)}
        res = CF.replay_counterfactual(record, divergence_turn=turn, substitute_action=int(action),
                                    our_policy=our_policy, opp_policy=opp_policy, opp_bot=opp_bot,
                                    post_t_seeds=post_seeds, stall_sides=stall_sides, text=narrate, core=core)
    finally:
        if own_core:
            core.close()
    root_tok = {int(k): v for k, v in res["root"]["tokens"].items()}.get(int(action))
    if root_tok != substitute_choice:
        raise RuntimeError(f"the core's root offers {root_tok!r} for action {action}, the walk {substitute_choice!r} "
                           "— the play-out root is not the decision the trace names")

    outcomes: List[str] = []
    win_traj = loss_traj = None      # the first winning / first losing rollout's play-by-play (narrate)
    for r in res["rollouts"]:
        outcomes.append(r.outcome)
        if narrate and r.text is not None:
            if r.outcome == "win" and win_traj is None:
                win_traj = CF.summarize_trajectory(our_side, [(our_side, "\n".join(r.text))])
            elif r.outcome == "loss" and loss_traj is None:
                loss_traj = CF.summarize_trajectory(our_side, [(our_side, "\n".join(r.text))])

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
    if spec["kind"] == "bot":
        caveats.append("Opponent = the in-core Rust port of the bot, its random streams SEEDED from this "
                       "battle + decision — the same distribution the eval played, not the same draws "
                       "(the eval's bot ran on its own episode streams).")
    if impl != "rust":
        caveats.append(f"--impl {impl} does not apply: the play-out runs on the in-process Rust core "
                       "(its only engine since the poke-env retirement's P6).")

    return {
        "inv": inv_index,
        "turn": turn,
        "trainee_side": our_side,
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
        "engine": ENGINE,
        "play_out": {"schema": CF.CF_SCHEMA, "opponent_recorded_choice": res["root"].get("other_recorded"),
                     "capped": sum(1 for r in res["rollouts"] if r.capped)},
    }
