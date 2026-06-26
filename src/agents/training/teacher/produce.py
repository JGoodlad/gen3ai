"""``produce_correction`` — turn ONE candidate into a VERIFIED-better :class:`Correction`, or skip it.

The 3-tier strictly-better gate (§4), all reusing the prober's better-line + rollout-confirm:
1. SEARCH (``session.better_line``) finds the best alternative A* vs the EXACT reloaded opponent;
2. CONFIRM rolls A* to the end vs that same exact opponent (Wilson CI);
3. GATE: keep only if the confirm's Wilson LOWER bound beats the played (loss) line — i.e. *verified*
   strictly better, not the critic's optimistic backed-up value (the Spore 95%-vs-62% lesson).

The AWR advantage is the CONFIRMED win-rate improvement (``confirmed − played_rate``), never a critic
advantage (the soundness point). An ``'unresolved'`` opponent is SKIPPED, never approximated.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from agents.training.teacher.buffer import Correction


def produce_correction(
    session,                       # ProbeSession with ckpt_override = the FROZEN trainee snapshot
    candidate,                     # teacher.selection.Candidate
    *,
    opponent_ckpt: Optional[str],
    opponent_source: str,          # 'ckpt' | 'bot' | 'unresolved'
    confirm_rollouts: int = 8,
    depth: int = 2, beam: int = 3, top_k: int = 4,
    margin_min: float = 0.0,
    search_session=None,           # injected WARM SearchSession (perf — amortizes the Node spawn)
) -> "Tuple[Optional[Correction], str]":
    """Returns ``(Correction, 'ok')`` for a verified-better correction, else ``(None, reason)`` where
    reason ∈ {unresolved, no_better_line, already_known, confirm_failed, gate_failed, error}."""
    if opponent_source == "unresolved":
        return None, "unresolved"          # never fall back to the trainee proxy (soundness)
    interior = "ckpt" if opponent_source == "ckpt" else "none"   # bots → recorded@divergence interior
    try:
        out = session.better_line(
            candidate.summary_path, candidate.inv_index, depth=depth, beam=beam, top_k=top_k,
            interior_opponent=interior, opponent_ckpt=opponent_ckpt,
            confirm_rollouts=confirm_rollouts, search_session=search_session)
    except (FileNotFoundError, RuntimeError, ValueError, IndexError) as e:
        return None, f"error:{type(e).__name__}"

    ba = out.get("best_alternative")
    if ba is None or ba.get("action") is None:
        return None, "no_better_line"
    a_star = int(ba["action"])

    # The recorded obs + the legal mask at the searched state (the AWR training target's state). obs is
    # the policy's own obs the trace saved; the mask is reconstructed from the candidates' legal actions.
    b = session._battle(candidate.summary_path)
    npz = session._npz(b)
    obs = np.asarray(npz["obs"][candidate.inv_index], dtype=np.float32)
    mask = np.zeros(11, dtype=np.int8)
    for c in out.get("candidates") or []:
        if c.get("action") is not None and 0 <= int(c["action"]) < 11:
            mask[int(c["action"])] = 1

    # Staleness re-verify (§9): if the FROZEN trainee already greedily plays A*, there's nothing to teach.
    try:
        model, _ = session._model_for(b)
        if int(np.argmax(model.action_probs_batch(obs[None], mask[None])[0])) == a_star:
            return None, "already_known"
    except Exception:  # noqa: BLE001 — the staleness check is best-effort; never block a real correction
        pass

    conf = out.get("confirm")
    if not conf or "error" in conf or conf.get("win_rate") is None:
        return None, "confirm_failed"
    ci = conf.get("ci") or [None, None]
    wilson_low = ci[0]
    played_rate = 0.0                       # a loss trace's realized rate (the crater + its window lost)
    if wilson_low is None or wilson_low <= played_rate + margin_min:
        return None, "gate_failed"          # NOT verified strictly-better → don't distill

    advantage = float(conf["win_rate"]) - played_rate    # the CONFIRMED improvement (the AWR weight)
    corr = Correction(
        obs=obs, action_mask=mask, better_action=a_star, advantage=advantage,
        confirmed_value=float(conf["win_rate"]),   # only the off-by-default value term reads this
        step_produced=int(candidate.step), opponent=candidate.opponent)
    return corr, "ok"
