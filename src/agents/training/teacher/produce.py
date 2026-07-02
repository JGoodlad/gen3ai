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


def _build_pi_target(out: dict, mask: np.ndarray, opd_beta: float) -> "Optional[np.ndarray]":
    """Build the OPD improved distribution π' over the 11 action slots from the beam's per-action
    backed-up values (``out['candidates']``). Softmax over LEGAL actions of ``(v(a) − max_legal_v) /
    opd_beta`` (temperature); a legal-but-unsearched slot gets a COMPLETED-Q floor (the min legal value)
    so its mass is small but non-zero; 0 on illegal slots; L1-normalized over legal. Returns ``None`` if
    no legal action carries a usable value (nothing to distill toward)."""
    legal = [a for a in range(11) if mask[a]]
    if not legal:
        return None
    # Per-action backed-up value v(a): prefer the search's `backup` (max-over-our-continuations), fall
    # back to the immediate `value`; a terminal `backup` is None (|backup| ≥ _WIN) so map win/loss to a
    # large ±sentinel via `terminal`. The played/recorded action's value anchors an unsearched slot.
    vals: dict = {}
    for c in out.get("candidates") or []:
        a = c.get("action")
        if a is None or not (0 <= int(a) < 11) or not mask[int(a)]:
            continue
        term = c.get("terminal")
        if term == "win":
            v = 1e3
        elif term == "loss":
            v = -1e3
        elif term == "tie":
            v = 0.0
        elif c.get("backup") is not None:
            v = float(c["backup"])
        elif c.get("value") is not None:
            v = float(c["value"])
        else:
            continue
        vals[int(a)] = v
    if not vals:
        return None
    # COMPLETED-Q floor: a legal action the beam didn't score gets the MIN searched legal value (a
    # pessimistic completion — it's plausibly no better than the worst we looked at), so it keeps a small
    # non-zero probability rather than an undefined one. The recorded/baseline value backstops if lower.
    baseline = out.get("baseline_value")
    if baseline is None:
        baseline = out.get("recorded_value")
    floor = min(list(vals.values()) + ([float(baseline)] if baseline is not None else []))
    for a in legal:
        vals.setdefault(a, floor)

    beta = opd_beta if opd_beta and opd_beta > 0 else 1.0
    v_arr = np.array([vals[a] for a in legal], dtype=np.float64)
    z = (v_arr - v_arr.max()) / beta          # subtract max for numerical stability
    w = np.exp(z)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        return None
    pi = np.zeros(11, dtype=np.float32)
    pi[legal] = (w / w_sum).astype(np.float32)  # already L1-normalized over legal (0 on illegal)
    return pi


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
    build_pi_target: bool = False, # OPD: also build the improved distribution π' (KL target) from the beam
    opd_beta: float = 1.0,         # OPD softmax temperature over the per-action backed-up values
) -> "Tuple[Optional[Correction], str]":
    """Returns ``(Correction, 'ok')`` for a verified-better correction, else ``(None, reason)`` where
    reason ∈ {unresolved, no_better_line, already_known, confirm_failed, gate_failed, error}.

    When ``build_pi_target`` (OPD requested), the returned Correction ALSO carries ``pi_target`` — the
    full improved distribution π' over the 11 actions — alongside the AWR fields (A*, advantage), so one
    run can A/B AWR vs KL. No cost when off (π' is only computed on the OPD path)."""
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
    if build_pi_target:                            # OPD: attach the full improved distribution π'
        corr.pi_target = _build_pi_target(out, mask, opd_beta)
    return corr, "ok"
