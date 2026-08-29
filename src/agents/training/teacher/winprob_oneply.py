"""`--search-teacher-mode winprob_oneply` — ONE-PLY WIN-PROB RANKING TARGETS (ai_v12 routes 2 + 3).

Design: `designs/ai_v12/design_winprob_behavior_coupling.md`. **Nothing has run this; no arm is
registered.** The default mode is `crater` and this module is not reached unless it is asked for.

WHAT IT PRODUCES, and how it differs from the `crater` mode beside it. `selection.py`/`produce.py`
answer *"where did the model lose the most value, and is there a strictly better LINE"* — a depth-2
beam over the CRITIC, gated by a Wilson CI on rollout-confirmed win rate. This module answers a
different question: *"at a decision the model's own win-probability head thinks is CONTESTED, does a
one-ply successor read prefer some other action, by a margin that survives confirmation?"*

The pipeline, which is the design doc's "3 filters → 2 transplants" as code:

  1. **CONTESTED gate** (route 3, filter 1) — ``n_legal >= 2`` AND ``|P(win|s) - 0.5| < band``. This
     is the H rule, and it is imported from `main.search_dividend.defensive.gate` rather than
     re-typed: two definitions of "contested" that could drift apart while both looked right is a
     failure this tree has already paid for elsewhere. Model-FREE — it reads the recorded
     ``win_probs`` / ``action_mask`` arrays the trace already carries, so a whole run's traces can be
     scanned without loading a checkpoint.
  2. **ONE-PLY read** (filter 2) — `ProbeSession.lookahead` re-rolls the turn under each legal action
     (the opponent plays its RECORDED move), materializes the successor through the real encoder, and
     reads the loaded model's heads. We take the WIN-PROB read, not V: the critic estimates SHAPED
     return in PopArt-normalized units, and probe G measured the win-prob head beating the played
     action on exactly this job.
  3. **MARGIN gate** (filter 3) — the preferred action must beat the played one by at least
     ``margin_min`` in win probability.
  4. **CONFIRMATION** (route 3 proper) — ``confirm_rollouts`` PAIRED rollouts to a terminal for A*
     and for the played action, via `ProbeSession.replay_counterfactual`. A rollout contains the
     opponent response the one-ply leaf structurally lacks, and A* must still win on the Wilson
     LOWER bound.
  5. **TRANSPLANT** (route 2) — the survivor becomes a `Correction` with the SAME fields the existing
     AWR loss already consumes, so nothing downstream of the ring buffer changed at all.

⚠️ **WHY STEP 4 IS A REQUIREMENT AND NOT A REFINEMENT — the WINNER'S CURSE.** The ranking instrument
is biased, and its bias is DIFFERENTIAL, which is exactly the quantity that matters here. Defensive
search iter 2 (`designs/research_state/measurements/defensive_search_iter2_2026-08-29.md`) un-throttled
its allocator, produced **13× more evidence-certified overrules (1.8% → 5.82%)** and landed the win
rate on **0.5003 [0.4803, 0.5203] — the point estimate IS the null**. CRN pairing removes dice noise
and the shared offset, so what a separation procedure CERTIFIES is the leaf's residual differential
bias (RMS 0.122, larger than most true gaps) as much as it is signal. **Statistical separation of a
biased reader is not correctness.** A distillation target has no invariance shield to fall back on —
unlike PBRS (route 1), a wrong target simply trains the policy to be wrong. So an un-confirmed
one-ply preference is not a cheaper version of this mode; it is that failure with a gradient attached.
``confirm_rollouts=0`` is available ONLY because the design doc's E2 needs it as the deliberately
undisciplined control arm.

The counter-evidence that keeps this mode alive rather than killing it: probe K re-judged iter 2's
3,531 overrules under opponent-MARGINALIZED ground truth and found **+0.0474 [+0.0216, +0.0730] per
decision — REAL**. The overrules were right; the per-decision → per-episode TRANSFER failed. A
training target changes the policy everywhere the network generalizes, not only at the ~2.2 decisions
per game where a searcher intervened — which is why the response to probe K is route 2, not a fourth
iteration of route 3 as an inference lever.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

from agents.training.teacher.buffer import Correction
from agents.training.teacher.selection import Candidate, _recon_for

#: The contested band's default half-width. This is `search_dividend.defensive.DEFAULT_WP_MARGIN`
#: (0.15), the operating point probe H measured — imported rather than re-typed so the teacher's
#: notion of "contested" and the searcher's cannot drift apart.
from main.search_dividend.defensive import DEFAULT_WP_MARGIN, GATE_SEARCH, DefensiveConfig, gate

#: The default one-ply Δφ a preference must clear. A WORKING default, not the measured floor: the
#: differential-bias RMS from defensive-search iter 2 is 0.122, which is larger than most true gaps
#: and would collapse target volume by roughly an order of magnitude. The design doc's E4 is the arm
#: that asks which of the two is the right number; E2 runs at this one.
DEFAULT_MARGIN_MIN = 0.02

#: Statuses `produce_winprob_correction` can return, so a caller can histogram them the way the
#: existing worker histograms `produce_correction`'s.
STATUSES = ("ok", "no_lookahead", "no_alternative", "margin_failed", "already_known",
            "confirm_failed", "gate_failed", "unresolved", "error")


# ──────────────────────────────────────────────────────────────────────────────────────────────
# PURE — the gates. No session, no model, no I/O. This is what the tests pin.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def is_contested(n_legal: int, win_prob: Optional[float], band: float = DEFAULT_WP_MARGIN) -> bool:
    """The H rule: at least two legal actions AND the head is not already sure of the outcome.

    Delegates to `search_dividend.defensive.gate`, which owns both clauses. A `None` win probability
    means the trace carries no head read at this decision (`--win-prob-mode none`, or an
    uncaptured row) — NOT contested, and never imputed: a decision we cannot judge is one we do not
    teach from.
    """
    if win_prob is None or (isinstance(win_prob, float) and math.isnan(win_prob)):
        return False
    return gate(int(n_legal), float(win_prob), DefensiveConfig(wp_margin=float(band))) == GATE_SEARCH


def rank_by_win_prob(candidates: "list[dict]") -> "list[tuple[int, float]]":
    """``lookahead`` candidate dicts → ``[(action, P(win | s'))]`` sorted best-first.

    Skips any candidate with no win-prob read: the whole point of this mode is the win-prob leaf, so
    falling back to `value` would silently produce a DIFFERENT teacher (the critic's shaped-return
    ranking) under the same flag — the exact confusion `defensive.check_leaf` exists to prevent.
    """
    out = []
    for c in candidates or []:
        a, wp = c.get("action"), c.get("win_prob")
        if a is None or wp is None:
            continue
        try:
            wp_f = float(wp)
        except (TypeError, ValueError):
            continue
        if math.isnan(wp_f):
            continue
        out.append((int(a), wp_f))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def clears_margin(ranked: "list[tuple[int, float]]", played_action: Optional[int],
                  margin_min: float) -> "Tuple[Optional[int], float]":
    """``(A*, margin)`` if the best-ranked action beats the PLAYED one by ``margin_min``, else
    ``(None, margin)``.

    The comparison is against the played action specifically, not against the runner-up: the target
    exists to move probability OFF what the policy did and ONTO something better, so "how much better
    than the second-best alternative" is not the quantity being gated.
    """
    if not ranked:
        return None, 0.0
    best_a, best_wp = ranked[0]
    played_wp = next((wp for a, wp in ranked if a == played_action), None)
    if played_wp is None:
        return None, 0.0                    # the played action was not scored — nothing to contrast
    margin = best_wp - played_wp
    if best_a == played_action:
        return None, margin                 # the policy already plays the head's preference
    if margin < float(margin_min):
        return None, margin
    return best_a, margin


def wilson_lower(wins: float, n: int, z: float = 1.96) -> float:
    """Wilson score interval's LOWER bound for a binomial rate. Mirrors the existing teacher gate's
    use of the prober's CI, expressed here so the paired-rollout arithmetic is testable without a
    session. ``n <= 0`` ⇒ 0.0 (no evidence is not evidence of superiority)."""
    if n <= 0:
        return 0.0
    p = float(wins) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(max(0.0, p * (1 - p) / n + z2 / (4 * n * n)))
    return max(0.0, (centre - margin) / denom)


def confirmed_better(star_wins: float, played_wins: float, n: int,
                     margin_min: float = 0.0) -> "Tuple[bool, float]":
    """Route 3's verdict on N PAIRED rollouts: ``(is_confirmed, advantage)``.

    ``advantage`` is the CONFIRMED win-rate improvement (A*'s realized rate minus the played
    action's) — the AWR weight, and never a critic advantage, which is the same soundness point
    `produce_correction` makes. The GATE is on A*'s Wilson LOWER bound vs the played action's POINT
    rate: an asymmetric test on purpose, because the failure this filter exists to catch is a
    flattering estimate of A*, not an unflattering one of what was played.
    """
    if n <= 0:
        return False, 0.0
    adv = (float(star_wins) - float(played_wins)) / float(n)
    ok = wilson_lower(star_wins, n) > (float(played_wins) / float(n)) + float(margin_min)
    return ok, adv


# ──────────────────────────────────────────────────────────────────────────────────────────────
# SELECTION — model-free, from the recorded arrays alone.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def select_winprob_candidates(
    run_dir: str,
    *,
    budget: int = 200,
    step: Optional[int] = None,
    band: float = DEFAULT_WP_MARGIN,
    outcome: Optional[str] = None,
    max_per_battle: int = 4,
    session=None,
) -> List[Candidate]:
    """Rank up to ``budget`` CONTESTED ``move_selection`` decisions from a run's traces.

    Unlike `selection.select_candidates` this does NOT filter to losses. A whiff in a won game is
    still a whiff, and the head's self-referential labels are exactly why it cannot be trusted to
    have noticed (a habitual mistake that still wins 55% teaches it "55%"). ``outcome`` is available
    for an arm that wants the loss-only slice, and defaults to every battle.

    Ranked by CONTESTEDNESS — ``|P(win) - 0.5|`` ascending — because that is the only ordering signal
    available before the (expensive) one-ply read. ``max_per_battle`` spreads the budget across
    battles rather than letting one long game with a flat mid-game fill it.

    Model-free: reads ``npz["win_probs"]`` and ``npz["action_mask"]``, both recorded at play time.
    """
    from main.prober.session import ProbeSession
    own = session is None
    sess = session if session is not None else ProbeSession(run_dir)
    try:
        if step is None:
            steps = [sg.step for sg in sess.tree.steps]
            step = max(steps) if steps else None
        rows = sess.battles(outcome=outcome, step=step)
        out: List[Candidate] = []
        for row in rows:
            summary_path = row["id"]
            if not row.get("has_npz") or not os.path.exists(_recon_for(summary_path)):
                continue                     # lookahead + replay need the reconstruction sibling
            try:
                b = sess._battle(summary_path)
                summary, npz = sess._summary(b), sess._npz(b)
                wps = np.asarray(npz["win_probs"], dtype=np.float64)
                masks = np.asarray(npz["action_mask"])
            except (KeyError, FileNotFoundError, ValueError):
                continue                     # a run with no win-prob head simply yields nothing
            picked = 0
            for i, inv in enumerate(summary.get("invocations", [])):
                if picked >= max_per_battle:
                    break
                if inv.get("phase") != "move_selection" or i >= len(wps) or i >= len(masks):
                    continue
                wp = float(wps[i])
                if not is_contested(int(np.asarray(masks[i]).sum()), wp, band):
                    continue
                out.append(Candidate(
                    summary_path=summary_path, recon_path=_recon_for(summary_path),
                    inv_index=i, turn=int(inv.get("turn", 0)),
                    opponent=row.get("opponent", "?"), step=int(row.get("step", step or 0)),
                    # NOT a value crater: the priority is CONTESTEDNESS, stored in the same slot so
                    # the Candidate record stays one shape across both teacher modes. Higher = more
                    # contested, so the existing descending sort is still the right one.
                    anchor_delta=float(band - abs(wp - 0.5)),
                    verdict="CONTESTED"))
                picked += 1
        out.sort(key=lambda c: c.anchor_delta, reverse=True)
        return out[:budget]
    finally:
        if own:
            try:
                sess.close()
            except Exception:  # noqa: BLE001 — never let teardown mask a real selection result
                pass


# ──────────────────────────────────────────────────────────────────────────────────────────────
# PRODUCTION — one candidate → a confirmed Correction, or a reason it was skipped.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _played_action(summary: dict, inv_index: int) -> Optional[int]:
    inv = summary.get("invocations", [])[inv_index] if inv_index < len(
        summary.get("invocations", [])) else None
    if not inv:
        return None
    a = inv.get("action_index", inv.get("action"))
    return int(a) if a is not None else None


def produce_winprob_correction(
    session,                        # ProbeSession pinned to the FROZEN trainee (ckpt_override=…)
    candidate: Candidate,
    *,
    opponent_ckpt: Optional[str],
    opponent_source: str,           # 'ckpt' | 'bot' | 'unresolved'
    margin_min: float = DEFAULT_MARGIN_MIN,
    confirm_rollouts: int = 8,
) -> "Tuple[Optional[Correction], str]":
    """One contested decision → a CONFIRMED `Correction`, or ``(None, reason)``.

    ``confirm_rollouts == 0`` SKIPS route 3's confirmation and takes the one-ply margin as the
    advantage. That is the design doc's E2 control arm and nothing else — see this module's docstring
    on why an un-confirmed target is the iter-2 failure with a gradient attached.
    """
    if opponent_source == "unresolved":
        return None, "unresolved"           # never approximate the opponent (the soundness rule)

    try:
        out = session.lookahead(candidate.summary_path, inv=candidate.inv_index)
    except (FileNotFoundError, RuntimeError, ValueError, IndexError, KeyError) as e:
        return None, f"error:{type(e).__name__}"
    cands = out.get("candidates") or []
    ranked = rank_by_win_prob(cands)
    if not ranked:
        return None, "no_lookahead"         # no win-prob leaf ⇒ this mode has nothing to rank

    b = session._battle(candidate.summary_path)
    summary, npz = session._summary(b), session._npz(b)
    played = _played_action(summary, candidate.inv_index)
    if played is None:
        played = int(np.asarray(npz["actions"])[candidate.inv_index])

    a_star, margin = clears_margin(ranked, played, margin_min)
    if a_star is None:
        return None, ("no_alternative" if margin <= 0.0 else "margin_failed")

    obs = np.asarray(npz["obs"][candidate.inv_index], dtype=np.float32)
    mask = np.asarray(npz["action_mask"][candidate.inv_index]).astype(np.int8).reshape(-1)[:11]
    if mask.shape[0] < 11:
        mask = np.pad(mask, (0, 11 - mask.shape[0]))

    # Staleness re-verify (the same check `produce_correction` makes): if the FROZEN trainee already
    # greedily plays A*, there is nothing to teach and the row would only sharpen an existing peak.
    try:
        model, _ = session._model_for(b)
        if int(np.argmax(model.action_probs_batch(obs[None], mask[None])[0])) == a_star:
            return None, "already_known"
    except Exception:  # noqa: BLE001 — best-effort; never block a real correction on a probe failure
        pass

    advantage = float(margin)
    if confirm_rollouts and int(confirm_rollouts) > 0:
        n = int(confirm_rollouts)
        try:
            star = session.replay_counterfactual(
                candidate.summary_path, candidate.inv_index, a_star,
                n_rollouts=n, opponent_ckpt=opponent_ckpt,
                opponent_source=("ckpt" if opponent_source == "ckpt" else "auto"))
            base = session.replay_counterfactual(
                candidate.summary_path, candidate.inv_index, played,
                n_rollouts=n, opponent_ckpt=opponent_ckpt,
                opponent_source=("ckpt" if opponent_source == "ckpt" else "auto"))
        except (FileNotFoundError, RuntimeError, ValueError, IndexError, KeyError) as e:
            return None, f"error:{type(e).__name__}"
        sr, br = star.get("win_rate"), base.get("win_rate")
        if sr is None or br is None:
            return None, "confirm_failed"
        ok, adv = confirmed_better(float(sr) * n, float(br) * n, n)
        if not ok:
            return None, "gate_failed"      # certified by the leaf, REFUSED by the rollouts
        advantage = float(adv)

    return Correction(
        obs=obs, action_mask=mask, better_action=int(a_star), advantage=advantage,
        confirmed_value=float(ranked[0][1]),   # read only by the off-by-default value term
        step_produced=int(candidate.step), opponent=candidate.opponent), "ok"
